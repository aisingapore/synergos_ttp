#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
from pathlib import Path

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
from rest_rpc.training.models import input_model
from rest_rpc.evaluation.core.server import start_proc
from rest_rpc.evaluation.core.utils import (
    ValidationRecords, 
    MLFRecords, 
    MLFlogger
)

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "validations", 
    description='API to faciliate model validation in a REST-RPC Grid.'
)

SUBJECT = "Validation"

db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
mlf_records = MLFRecords(db_path=db_path)
participant_records = ParticipantRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
tag_records = TagRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

mlflow_dir = app.config['MLFLOW_DIR']
mlf_logger = MLFlogger()

################################################################
# Validations - Used for marshalling (i.e. moulding responses) #
################################################################

# Marshalling inputs 
# - same input_model retrieved from the Models resource

# Marshalling Outputs
stats_model = ns_api.model(
    name="statistics",
    model={
        'accuracy': fields.Float(),
        'roc_auc_score': fields.Float(),
        'pr_auc_score': fields.Float(),
        'f_score': fields.Float(),
        'TPR': fields.Float(),
        'TNR': fields.Float(),
        'PPV': fields.Float(),
        'NPV': fields.Float(),
        'FPR': fields.Float(),
        'FNR': fields.Float(),
        'FDR': fields.Float(),
        'TP': fields.Integer(),
        'TN': fields.Integer(),
        'FP': fields.Integer(),
        'FN': fields.Integer()
    }
)

meta_stats_model = ns_api.model(
    name="meta_statistics",
    model={
        'statistics': fields.Nested(stats_model, skip_none=True),
        'res_path': fields.String(skip_none=True)
    }
)

val_inferences_model = ns_api.model(
    name="validation_inferences",
    model={
        'evaluate': fields.Nested(meta_stats_model, required=True)
    }
)

val_output_model = ns_api.inherit(
    "validation_output",
    val_inferences_model,
    {
        'doc_id': fields.String(),
        'kind': fields.String(),
        'key': fields.Nested(
            ns_api.model(
                name='key',
                model={
                    'participant_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(SUBJECT, ns_api, val_output_model)

#############
# Resources #
#############

@ns_api.route('/', defaults={
    'expt_id': None, 
    'run_id': None, 
    'participant_id': None
})
@ns_api.route('/<expt_id>', defaults={'run_id': None, 'participant_id': None})
@ns_api.route('/<expt_id>/<run_id>', defaults={'participant_id': None})
@ns_api.route('/<expt_id>/<run_id>/<participant_id>')
@ns_api.response(404, 'Validations not found')
@ns_api.response(500, 'Internal failure')
class Validations(Resource):
    """ Handles model inference within the PySyft grid. Model validation is done
        a series of X stages:
        1) Specified model architectures are re-loaded alongside its run configs
        2) Federated grid is re-established
        3) Inference is performed on participant's validation datasets
        4) `worker/predict` route is activated to export results to worker node
        5) Statistics are computed on the remote node
        6) Statistics are returned as response to the TTP and archived
        7) Log all polled statistics in MLFlow
        8) Repeat 1-7 for each worker, for all models
    """
    
    @ns_api.doc("get_validations")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, project_id, expt_id, run_id, participant_id):
        """ Retrieves validation statistics corresponding to experiment and run 
            parameters for a specified project
        """
        filter = {k:v for k,v in request.view_args.items() if v is not None}

        retrieved_validations = registration_records.read_all(filter=filter)
        
        if retrieved_validations:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="validations.get",
                params={
                    'project_id': project_id, 
                    'expt_id': expt_id,
                    'run_id': run_id    
                },
                data=retrieved_validations
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Validations do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_predictions")
    @ns_api.expect(input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, project_id, expt_id, run_id, participant_id):
        """ Triggers FL inference for specified project-experiment-run
            combination within a PySyft FL grid. 
            Note: Participants have the option to specify additional datasets
                  here for prediction. However all prediction sets will be
                  collated into a single combined prediction set to facilitate
                  prediction. Also, all prediction tags submitted here will
                  override any previously submitted prediction tags registered
                  for a specified project. This is to prevent accumulation of 
                  unsynced dataset tags (w.r.t participant's node). Hence, if
                  a participant wants to expand their testing datasets by 
                  declaring new tags, the onus is on them to declare all of it
                  per submission.

            Sample payload:

            {
                "dockerised": true
            }
        """
        # Populate grid-initialising parameters
        init_params = request.json

        # Retrieves expt-run supersets (i.e. before filtering for relevancy)
        retrieved_project = project_records.read(project_id=project_id)
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

        # Retrieve all relevant participant IDs, collapsing evaluation space if
        # a specific participant was declared
        participants = [
            record['participant']['id'] 
            for record in registrations
        ] if not participant_id else [participant_id]

        # Template for starting FL grid and initialising validation
        kwargs = {
            'experiments': experiments,
            'runs': runs,
            'registrations': registrations,
            'participants': participants,
            'metas': ['evaluate'],
            'version': None # defaults to final state of federated grid
        }
        kwargs.update(init_params)

        completed_validations = start_proc({project_id: kwargs})

        retrieved_validations = []
        for combination_key, validation_stats in completed_validations.items():

            # Store output metadata into database
            for participant_id, inference_stats in validation_stats.items():

                worker_key = (participant_id,) + combination_key

                new_validation = validation_records.create(
                    *worker_key,
                    details=inference_stats
                )

                retrieved_validation = validation_records.read(*worker_key)

                assert new_validation.doc_id == retrieved_validation.doc_id
                retrieved_validations.append(retrieved_validation)

        # Log all statistics to MLFlow
        mlf_logger.log(accumulations=completed_validations)

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="validations.post",
            params=request.view_args,
            data=retrieved_validations
        )
        return success_payload, 200