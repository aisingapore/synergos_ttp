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
from rest_rpc.training.core.utils import (
    AlignmentRecords, 
    ModelRecords,
    Poller,
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
    name="training_input",
    model={
        'auto_align': fields.Boolean(default=True, required=True),
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
model_meta_model = ns_api.model(
    name="model_meta",
    model={
        'origin': fields.String(required=True),
        'path': fields.String(required=True),
        'loss_history': fields.String(required=True)
    }
)

local_model_field = fields.Wildcard(fields.Nested(model_meta_model))
model_model = ns_api.model(
    name="model",
    model={
        'global': fields.Nested(model_meta_model, required=True),
        'local_*': local_model_field
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
        # ,
        # 'relations': fields.Nested(
        #     ns_api.model(
        #         name='model_relations',
        #         model={
        #             'Validation': fields.List(
        #                 fields.Nested(expt_output_model, skip_none=True)
        #             ),
        #             'Prediction': fields.List(
        #                 fields.Nested(run_output_model, skip_none=True)
        #             )
        #         }
        #     ),
        #     default={},
        #     required=True
        # )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, model_output_model)

#############
# Resources #
#############

# Insert Models representation here for mass automation

@ns_api.route('/', defaults={'expt_id': None, 'run_id': None})
@ns_api.route('/<expt_id>', defaults={'run_id': None})
@ns_api.route('/<expt_id>/<run_id>')
@ns_api.response(404, 'model not found')
@ns_api.response(500, 'Internal failure')
class Models(Resource):
    """ Handles model training within the PySyft grid. Since model training is
        deterministic, there will NOT be a resource to cater to a collection of 
        models 
    """
    
    @ns_api.doc("get_models")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, project_id, expt_id, run_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        filter = {k:v for k,v in request.view_args.items() if v is not None}

        retrieved_models = model_records.read_all(filter=filter)
        
        if retrieved_models:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="models.get",
                params={
                    'project_id': project_id, 
                    'expt_id': expt_id,
                    'run_id': run_id    
                },
                data=retrieved_models
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Models does not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_training")
    @ns_api.expect(input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, project_id, expt_id, run_id):
        """ Triggers FL training for specified experiment & run parameters by
            initialising a PySyft FL grid
        """
        # Populate grid-initialising parameters
        init_params = request.json

        # Retrieves expt-run supersets (i.e. before filtering for relevancy)
        retrieved_project = project_records.read(project_id=project_id)
        project_action = retrieved_project['action']
        experiments = retrieved_project['relations']['Experiment']
        runs = retrieved_project['relations']['Run']

        # If specific experiment was declared, collapse training space
        if expt_id:

            retrieved_expt = expt_records.read(
                project_id=project_id, 
                expt_id=expt_id
            )
            runs = retrieved_expt.pop('relations')['Run']
            experiments = [retrieved_expt]

            # If specific run was declared, further collapse training space
            if run_id:

                retrieved_run = run_records.read(
                    project_id=project_id, 
                    expt_id=expt_id,
                    run_id=run_id
                )
                retrieved_run.pop('relations')
                runs = [retrieved_run]

        # Retrieve all participants' metadata
        registrations = registration_records.read_all(
            filter={'project_id': project_id}
        )

        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # Decoupling of MFA from training cycle is required. With this, polling is
        # skipped since alignment is not triggered

        # [Problems]
        # When alignment is not triggered, workers are not polled for their headers
        # and schemas. Since project logs are generated via polling, not doing so
        # results in an error for subsequent operations

        # [Solution]
        # Poll irregardless of alignment. Modify Worker's Poll endpoint to be able 
        # to handle repeated initiialisations (i.e. create project logs if it does
        # not exist, otherwise retrieve)

        auto_align = init_params['auto_align']
        if not auto_align:
            poller = Poller(project_id=project_id)
            poller.poll(registrations)

        # Template for starting FL grid and initialising training
        kwargs = {
            'action': project_action,
            'experiments': experiments,
            'runs': runs,
            'registrations': registrations
        }
        kwargs.update(init_params)

        completed_trainings = start_proc(kwargs)

        # Store output metadata into database
        retrieved_models = []
        for (project_id, expt_id, run_id), data in completed_trainings.items():

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
            method="models.post",
            params=request.view_args,
            data=retrieved_models
        )
        return success_payload, 200
