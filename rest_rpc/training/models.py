#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import Dict, List, Union, Any

# Libs
from flask import request
from flask_restx import Namespace, Resource, fields
from tinydb.database import Document

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import TopicalPayload
from rest_rpc.training.core.utils import Poller, RPCFormatter
from rest_rpc.training.core.server import execute_combination_training
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
from synmanager.train_operations import TrainProducerOperator

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

ns_api = Namespace(
    "models", 
    description='API to faciliate model training in a PySyft Grid.'
)

grid_idx = app.config['GRID']

db_path = app.config['DB_PATH']
collab_records = CollaborationRecords(db_path=db_path)
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
participant_records = ParticipantRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
tag_records = TagRecords(db_path=db_path)
alignment_records = AlignmentRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("training/models.py logged", Description="No Changes")

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

payload_formatter = TopicalPayload(
    subject=model_records.subject, 
    namespace=ns_api, 
    model=model_output_model
)

########
# Jobs #
########

def execute_training_job(
    combination_key: List[str],
    combination_params: Dict[str, Union[str, int, float, list, dict]]
) -> List[Document]:
    """ Encapsulated job function to be compatible for queue integrations.
        Executes model training for a speciifed federated cycle, and stores all
        outputs for subsequent use.

    Args:
        grid (list(dict))): Registry of participants' node information
        combination_key (dict): Composite IDs of a federated combination
        combination_params (dict): Initializing parameters for a training job
    Returns:
        Trained models (list(Document))    
    """
    collab_id, project_id, _, _ = combination_key

    # Retrieve all participants' metadata
    registrations = registration_records.read_all(
        filter={'collab_id': collab_id, 'project_id': project_id}
    )
    usable_grids = rpc_formatter.extract_grids(registrations)
    selected_grid = usable_grids[grid_idx]

    training_data = execute_combination_training(
        grid=selected_grid,
        **combination_params
    ) 

    # Store output metadata into database
    model_records.create(*combination_key, details=training_data)
    retrieved_model = model_records.read(*combination_key)

    return retrieved_model

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
    def get(self, collab_id, project_id, expt_id, run_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        filter = {k:v for k,v in request.view_args.items() if v is not None}

        retrieved_models = model_records.read_all(filter=filter)
        
        if retrieved_models:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="models.get",
                params=request.view_args,
                data=retrieved_models
            )

            logging.info(
                "Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' > Models: Record(s) retrieval successful!".format(
                    collab_id, project_id, expt_id, run_id
                ),
                code=200, 
                description="Model(s) for specified federated combination(s) successfully retrieved!",
                ID_path=SOURCE_FILE,
                ID_class=Models.__name__, 
                ID_function=Models.get.__name__,
                **request.view_args
            )

            return success_payload, 200

        else:
            logging.error(
                "Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' > Models: Record(s) retrieval failed.".format(
                    collab_id, project_id, expt_id, run_id
                ),
                code=404,
                description="Model(s) does not exist for specified keyword filters!",
                ID_path=SOURCE_FILE,
                ID_class=Models.__name__, 
                ID_function=Models.get.__name__,
                **request.view_args
            )
            ns_api.abort(
                code=404, 
                message=f"Models does not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_training")
    @ns_api.expect(input_model)
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, collab_id, project_id, expt_id, run_id):
        """ Triggers FL training for specified experiment & run parameters by
            initialising a PySyft FL grid
        """
        init_params = request.json

        # Retrieve all connectivity settings for all Synergos components
        retrieved_collaboration = collab_records.read(collab_id=collab_id)

        # Retrieve expt-run supersets (i.e. before filtering for relevancy)
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

        # Extract all possible federated combinations given specified keys
        training_combinations = rpc_formatter.enumerate_federated_conbinations(
            action=project_action,
            experiments=experiments,
            runs=runs,
            **init_params
        )

        is_cluster = False
        if is_cluster:
            # Submit parameters of federated combinations to job queue
            queue_host = retrieved_collaboration['mq_host']
            queue_port = retrieved_collaboration['mq_port']
            train_producer = TrainProducerOperator(
                host=queue_host, 
                port=queue_port
            )
            for train_key, train_kwargs in training_combinations.items():
                train_producer.process({
                    'process': 'train',  # operations filter for MQ consumer
                    'combination_key': train_key,
                    'combination_params': train_kwargs
                })
            retrieved_models = []

        else:
            # Run federated combinations sequentially using selected grid
            retrieved_models = [
                execute_training_job(
                    combination_key=train_key, 
                    combination_params=train_kwargs
                )
                for train_key, train_kwargs in training_combinations.items()
            ]

        success_payload = payload_formatter.construct_success_payload(
            status=200,
            method="models.post",
            params=request.view_args,
            data=retrieved_models
        )
        logging.info(
            "Collaboration '{}' > Project '{}' > Experiment '{}' > Run '{}' > Models: Record(s) creation successful!.".format(
                collab_id, project_id, expt_id, run_id
            ),
            description="Model(s) for specified federated conditions were successfully created!",
            code=201, 
            ID_path=SOURCE_FILE,
            ID_class=Models.__name__, 
            ID_function=Models.post.__name__,
            **request.view_args
        )
            
        return success_payload, 200
