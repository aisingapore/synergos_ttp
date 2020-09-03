#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os

# Libs
import jsonschema
import mlflow
from flask import request
from flask_restx import Namespace, Resource, fields

# Custom
from rest_rpc import app
from rest_rpc.training.core.hypertuners.nni_interface import NNITuner
from rest_rpc.training.core.hypertuners.nni_driver_script import optim_prefix
from rest_rpc.connection.core.utils import TopicalPayload, RunRecords
from rest_rpc.evaluation.core.utils import ValidationRecords
from rest_rpc.evaluation.validations import val_output_model

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

ns_api = Namespace(
    "optimizations", 
    description='API to faciliate hyperparameter tuning in a federated grid.'
)

SUBJECT = "Optimization"

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

###########################################################
# Models - Used for marshalling (i.e. moulding responses) #
###########################################################

# Marshalling inputs
tuning_model = ns_api.model(
    name="tuning_input",
    model={
        'search_space': fields.Raw(required=True),
        'tuner': fields.String(),
        'optimize_mode': fields.String(),
        'trial_concurrency': fields.Integer(default=1),
        'max_exec_duration': fields.String(default="1h"),
        'max_trial_num': fields.Integer(default=10),
        'is_remote': fields.Boolean(default=True),
        'use_annotation': fields.Boolean(default=True),
        'dockerised': fields.Boolean(default=False, required=True),
        'verbose': fields.Boolean(default=False),
        'log_msgs': fields.Boolean(default=False)
    }
)

# Marshalling Outputs
# - same `val_output_model` retrieved from Validations resource

payload_formatter = TopicalPayload(SUBJECT, ns_api, val_output_model)

#############
# Resources #
#############

@ns_api.route('/')
@ns_api.response(404, 'Alignment not found')
@ns_api.response(500, 'Internal failure')
class Optimizations(Resource):
    """ Handles hyperparameter tuning  within the PySyft grid. This targets the
        specific experimental model for optimization given a user-defined search
        space and performs a full federated cycle within the scope of 
        hyperparameter ranges.
    """

    @ns_api.doc("get_optimizations")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def get(self, project_id, expt_id):
        """ Retrieves global model corresponding to experiment and run 
            parameters for a specified project
        """
        filter_keys = request.view_args

        retrieved_validations = validation_records.read_all(filter=filter_keys)
        optim_validations = [
            record 
            for record in retrieved_validations
            if optim_prefix in record['key']['run_id']
        ]
        
        if optim_validations:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="optimizations.get",
                params=request.view_args,
                data=optim_validations
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Optimizations do not exist for specified keyword filters!"
            )


    @ns_api.doc("trigger_optimizations")
    @ns_api.marshal_with(payload_formatter.plural_model)
    def post(self, project_id, expt_id):
        """ Creates sets of hyperparameters using a specified AutoML algorithm,
            within the scope of a user-specified search space, and conducts a 
            federated cycle for each proposed set. A federated cycle involves:
            1) Training model using specified experimental architecture on
               training data across the grid
            2) Validating trained model on validation data across the grid

            JSON received is expected to contain the following information:

            eg.

            {
                'search_space': {
                    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
                    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
                    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
                    "momentum":{"_type":"uniform","_value":[0, 1]}
                },
                'tuner': "TPE",
                'metric': "accuracy",
                'optimize_mode': "maximize",
                'trial_concurrency': 1,
                'max_exec_duration': "1h",
                'max_trial_num': 10,
                'is_remote': True,
                'use_annotation': True,
                'dockerised': True,
                'verbose': True,
                'log_msgs': True
            }
        """
        # Populate hyperparameter tuning parameters
        tuning_params = request.json

        # Create log directory
        optim_log_dir = os.path.join(out_dir, project_id, expt_id)

        nni_tuner = NNITuner(log_dir=optim_log_dir)
        nni_tuner.tune(**tuning_params)

        filter_keys = request.view_args
        search_space = tuning_params['search_space']
        retrieved_validations = validation_records.read_all(filter=filter_keys)
        optim_validations = []
        for record in retrieved_validations:

            # Check if validation record belongs to an optimization run
            curr_run_id = record['key']['run_id']
            if optim_prefix in curr_run_id:
                
                # Verify that current set of hyperparameters were indeed 
                # extracted from specified search space
                hyperparameters = run_records.read(
                    project_id=project_id,
                    expt_id=expt_id, 
                    run_id=curr_run_id
                )
                for param, value in hyperparameters.items():
                    tuning_type = search_space[param]['_type']
                    tuning_values = search_space[param]['_value']

                    if tuning_type == 'choice':
                        assert value in tuning_values

                    elif tuning_type == 'uniform':
                        assert value >= tuning_values[0]
                        assert value <= tuning_values[-1]
                
        if optim_validations:
            
            success_payload = payload_formatter.construct_success_payload(
                status=200,
                method="optimizations.get",
                params=request.view_args,
                data=optim_validations
            )
            return success_payload, 200

        else:
            ns_api.abort(
                code=404, 
                message=f"Optimizations do not exist for specified keyword filters!"
            )