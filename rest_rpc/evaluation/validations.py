#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from logging import NOTSET
from typing import Dict, List, Union, Any

# Libs
from flask import request
from flask_restx import Namespace, Resource, fields
from tinydb.database import Document

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.core.utils import RPCFormatter
from rest_rpc.evaluation.core.server import execute_combination_inference
from rest_rpc.evaluation.core.utils import MLFlogger
from synarchive.connection import (
    CollaborationRecords,
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    ParticipantRecords,
    RegistrationRecords,
    TagRecords
)
from synarchive.training import AlignmentRecords, ModelRecords
from synarchive.evaluation import ValidationRecords, MLFRecords
from synmanager.evaluate_operations import EvaluateProducerOperator

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "validations", 
    description='API to faciliate model validation in a REST-RPC Grid.'
)

grid_idx = app.config['GRID']

db_path = app.config['DB_PATH']
collab_records = CollaborationRecords(db_path=db_path)
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

rpc_formatter = RPCFormatter()

mlflow_dir = app.config['MLFLOW_DIR']
mlf_logger = MLFlogger()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/validations.py logged", Description="No Changes")

################################################################
# Validations - Used for marshalling (i.e. moulding responses) #
################################################################

# Marshalling inputs 
input_model = ns_api.model(
    name="validation_input",
    model={
        'auto_align': fields.Boolean(default=True, required=True),
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
stats_model = ns_api.model(
    name="statistics",
    model={
        'R2': fields.Float(),
        'MSE': fields.Float(),
        'MAE': fields.Float(),
        'accuracy': fields.List(fields.Float()),
        'roc_auc_score': fields.List(fields.Float()),
        'pr_auc_score': fields.List(fields.Float()),
        'f_score': fields.List(fields.Float()),
        'TPRs': fields.List(fields.Float()),
        'TNRs': fields.List(fields.Float()),
        'PPVs': fields.List(fields.Float()),
        'NPVs': fields.List(fields.Float()),
        'FPRs': fields.List(fields.Float()),
        'FNRs': fields.List(fields.Float()),
        'FDRs': fields.List(fields.Float()),
        'TPs': fields.List(fields.Integer()),
        'TNs': fields.List(fields.Integer()),
        'FPs': fields.List(fields.Integer()),
        'FNs': fields.List(fields.Integer())
    },
    skip_none=True
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
                    'collab_id': fields.String(),
                    'project_id': fields.String(),
                    'expt_id': fields.String(),
                    'run_id': fields.String()
                }
            ),
            required=True
        )
    }
)

payload_formatter = TopicalPayload(
    subject=validation_records.subject, 
    namespace=ns_api, 
    model=val_output_model
)

########
# Jobs #
########

def execute_validation_job(
    combination_key: List[str],
    combination_params: Dict[str, Union[str, int, float, list, dict]]
) -> List[Document]:
    """ Encapsulated job function to be compatible for queue integrations.
        Executes model validation for a specified federated cycle, and stores
        all outputs for subsequent use.

    Args:
        combination_key (dict): Composite IDs of a federated combination
        combination_params (dict): Initializing parameters for a validation job
    Returns:
        Validation statistics (list(Document))    
    """
    collab_id, project_id, _, _ = combination_key

    # Retrieve all participants' metadata
    registrations = registration_records.read_all(
        filter={'collab_id': collab_id, 'project_id': project_id}
    )
    usable_grids = rpc_formatter.extract_grids(registrations)
    selected_grid = usable_grids[grid_idx]

    completed_validations = execute_combination_inference(
        grid=selected_grid,
        **combination_params
    ) 

    # Store output metadata into database
    retrieved_validations = []
    for participant_id, inference_stats in completed_validations.items():
        
        # Log the inference stats
        worker_keys = (participant_id,) + combination_key
        validation_records.create(*worker_keys, details=inference_stats)
        retrieved_validation = validation_records.read(*worker_keys)
        retrieved_validations.append(retrieved_validation)

    # Log all statistics to MLFlow
    mlf_logger.log(accumulations={combination_key: completed_validations})

    return retrieved_validations

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
    def get(self, collab_id, project_id, expt_id, run_id, participant_id):
        """ Retrieves validation statistics corresponding to experiment and run 
            parameters for a specified project
        """
        filter = {k:v for k,v in request.view_args.items() if v is not None}

        retrieved_validations = validation_records.read_all(filter=filter)
        
        if retrieved_validations:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="validations.get",
                params=request.view_args,
                data=retrieved_validations
            )

            logging.info(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Validations: Bulk record retrieval successful!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                code=200, 
                description="Validations for participant '{}' under collaboration '{}''s project '{}' using experiment '{}' and run '{}' was successfully retrieved!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                ID_path=SOURCE_FILE,
                ID_class=Validations.__name__, 
                ID_function=Validations.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Validations: Bulk record retrieval failed!".format(
                    participant_id, collab_id, project_id, expt_id, run_id
                ),
                code=404, 
                description=f"Predictions do not exist for specified keyword filters!", 
                ID_path=SOURCE_FILE,
                ID_class=Validations.__name__, 
                ID_function=Validations.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Validations do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_predictions")
    @ns_api.expect(input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, collab_id, project_id, expt_id, run_id, participant_id):
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
                "auto_align": true,
                "dockerised": true
            }
        """
        init_params = request.json

        # Retrieve all connectivity settings for all Synergos components
        retrieved_collaboration = collab_records.read(collab_id=collab_id)

        # Retrieves expt-run supersets (i.e. before filtering for relevancy)
        retrieved_project = project_records.read(
            collab_id=collab_id,
            project_id=project_id
        )
        project_action = retrieved_project['action']
        experiments = retrieved_project['relations']['Experiment']
        runs = retrieved_project['relations']['Run']

        # If specific experiment was declared, collapse training space
        if expt_id:

            retrieved_expt = expt_records.read(
                collab_id=collab_id,
                project_id=project_id, 
                expt_id=expt_id
            )
            runs = retrieved_expt['relations']['Run']
            experiments = [retrieved_expt]

            # If specific run was declared, further collapse training space
            if run_id:

                retrieved_run = run_records.read(
                    collab_id=collab_id,
                    project_id=project_id, 
                    expt_id=expt_id,
                    run_id=run_id
                )
                runs = [retrieved_run]

        # Retrieve all relevant participant IDs, collapsing evaluation space if
        # a specific participant was declared
        registrations = registration_records.read_all(
            filter={'collab_id': collab_id, 'project_id': project_id}
        )
        participants = [
            record['participant']['id'] 
            for record in registrations
        ] if not participant_id else [participant_id]

        # Extract all possible federated combinations given specified keys
        valid_combinations = rpc_formatter.enumerate_federated_conbinations(
            action=project_action,
            experiments=experiments,
            runs=runs,
            participants=participants,
            metas=['evaluate'],
            version=None, # defaults to final state of federated grid
            **init_params
        )

        is_cluster = False
        if is_cluster:
            # Submit parameters of federated combinations to job queue
            queue_host = retrieved_collaboration['mq_host']
            queue_port = retrieved_collaboration['mq_port']
            valid_producer = EvaluateProducerOperator(
                host=queue_host, 
                port=queue_port
            )
            for valid_key, valid_kwargs in valid_combinations.items():
                valid_producer.process({
                    'process': 'validate',  # operations filter for MQ consumer
                    'combination_key': valid_key,
                    'combination_params': valid_kwargs
                })
            all_validations = []
        
        else:
            # Run federated combinations sequentially using selected grid
            all_validations = []
            for valid_key, valid_kwargs in valid_combinations.items():
                
                retrieved_combination_validations = execute_validation_job(
                    combination_key=valid_key, 
                    combination_params=valid_kwargs
                )

                # Flatten out list of predictions
                all_validations += retrieved_combination_validations

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="validations.post",
            params=request.view_args,
            data=all_validations
        )

        logging.info(
            "Participant '{}' >|< Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' >|< Validations: Record creation successful!".format(
                participant_id, collab_id, project_id, expt_id, run_id
            ),
            description=f"Validations for participant '{participant_id}' under project '{project_id}' using experiment '{expt_id}' and run '{run_id}' was successfully collected!",
            code=201, 
            ID_path=SOURCE_FILE,
            ID_class=Validations.__name__, 
            ID_function=Validations.post.__name__,
            **request.view_args
        )

        return success_payload, 200