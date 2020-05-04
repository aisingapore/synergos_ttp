#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging

# Libs
import jsonschema
import mlflow
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    TopicalPayload,
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    ParticipantRecords,
    RegistrationRecords,
    TagRecords
)
from rest_rpc.connection.projects import project_model
from rest_rpc.connection.experiments import expt_model
from rest_rpc.connection.runs import config_model
from rest_rpc.connection.participants import participant_model
from rest_rpc.connection.registration import registration_model
from rest_rpc.connection.tags import tag_model
from rest_rpc.training.core.utils import (
    AlignmentRecords, 
    ModelRecords,
    RPCFormatter
)
from rest_rpc.training.core.server import start_proc
from rest_rpc.training.alignments import alignment_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "models", 
    description='API to faciliate model training in a PySyft Grid.'
)

SUBJECT = "Model"

db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
participant_records = ParticipantRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
tag_records = TagRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling inputs
input_model = ns_api.model(
    name="input",
    model={
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
output_model = ns_api.model(
    name="output",
    model={
        'participants': fields.Nested(participant_model),
        'registration': fields.Nested(registration_model),
        'tags': fields.Nested(tag_model),
        'alignments': fields.Nested(alignment_model),
        'project': fields.Nested(project_model),
        'experiment': fields.Nested(expt_model),
        'run': fields.Nested(config_model)
    }
)

model_meta_model = ns_api.model(
    name="model_meta",
    model={
        'origin': fields.String(required=True),
        'path': fields.String(required=True)
    }
)

model_model = ns_api.model(
    name="model",
    model={
        "global": fields.Nested(model_meta_model, required=True)
    }
)

model_output_model = ns_api.inherit(
    "model_output",
    model_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, model_output_model)

#############
# Resources #
#############

# Insert Models representation here for mass automation

@ns_api.route('/')
@ns_api.response(404, 'model not found')
@ns_api.response(500, 'Internal failure')
class Models(Resource):
    """ Handles model training within the PySyft grid. Since model training is
        deterministic, there will NOT be a resource to cater to a collection of 
        models 
    """
    
    @ns_api.doc("get_model")
    #@ns_api.marshal_with(payload_formatter.singular_model)
    def get(self, project_id, expt_id, run_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        retrieved_model = model_records.read(
            project_id=project_id,
            expt_id=expt_id,
            run_id=run_id
        )
        
        if retrieved_model:
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="model.get",
                params={
                    'project_id': project_id, 
                    'expt_id': expt_id,
                    'run_id': run_id    
                },
                data=retrieved_model
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Model does not exist for Run {run_id} under Experiment {expt_id} under Project '{project_id}'!"
            )


    @ns_api.doc("trigger_single_training")
    @ns_api.expect(input_model)
    #@ns_api.marshal_with(payload_formatter.singular_model)
    def post(self, project_id, expt_id, run_id):
        """ Triggers FL training for specified experiment & run parameters by
            initialising a PySyft FL grid
        """
        init_params = request.json

        retrieved_expt = rpc_formatter.strip_keys(
            expt_records.read(
                project_id=project_id, 
                expt_id=expt_id
            ),
            concise=True
        )

        retrieved_run = rpc_formatter.strip_keys(
            run_records.read(
                project_id=project_id, 
                expt_id=expt_id,
                run_id=run_id
            ),
            concise=True
        )

        all_relevant_registrations = [
            rpc_formatter.strip_keys(record)
            for record in registration_records.read_all(
                filter={'project_id': project_id}
            ) 
        ]

        kwargs = {
            'keys': {
                'project_id': project_id,
                'expt_id': expt_id,
                'run_id': run_id
            },
            'experiments': {expt_id: retrieved_expt['model']},
            'runs': {run_id: retrieved_run},
            'registrations': all_relevant_registrations
        }
        kwargs.update(init_params)

        completed_trainings = start_proc(kwargs)

        # Store output metadata into database
        retrieved_models = []
        for (expt_id, run_id), data in completed_trainings.items():
            
            new_model = model_records.create(
                project_id=project_id,
                expt_id=expt_id,
                run_id=run_id,
                details=data
            )

            retrieved_model = model_records.read(
                project_id=project_id,
                expt_id=expt_id,
                run_id=run_id
            )

            assert new_model.doc_id == retrieved_model.doc_id
            retrieved_models.append(retrieved_model)

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="model.get",
            params={
                'project_id': project_id, 
                'expt_id': expt_id,
                'run_id': run_id    
            },
            data=retrieved_models
        )
        return success_payload, 200

