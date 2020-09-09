#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import argparse
import logging
import uuid
from string import Template

# Libs
import nni
from nni.utils import merge_parameter

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    RegistrationRecords,
    ExperimentRecords,
    RunRecords
)
from rest_rpc.training.core.utils import ModelRecords
from rest_rpc.evaluation.core.utils import ValidationRecords, MLFlogger
from rest_rpc.training.core.server import start_expt_run_training
from rest_rpc.evaluation.core.server import start_expt_run_inference

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

db_path = app.config['DB_PATH']
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

mlflow_dir = app.config['MLFLOW_DIR']
mlf_logger = MLFlogger()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

#############
# Functions #
#############

def main(
    project_id: str,
    expt_id: str,
    metric: str,
    dockerised: bool = True, 
    log_msgs: bool = True, 
    verbose: bool = True,
    **params
):
    """ Stores run parameters, train model on specified parameter set, and
        extract validation statistics on validation sets across the federated
        grid

    Args:
        project_id (str): Project ID of core project
        expt_id (str): Experiment ID of experimental model architecture
        metric (str): Statistical metric to optimise
        dockerised (bool): Toggles use of dockerised port orchestrations
        log_msgs (bool): Toggles if intermediary operations will be logged
        verbose (bool): Toggles if logging will be started in verbose mode
        **params: Hyperparameter set to train experiment model on
    """
    # Retrieve registered participants' metadata under specified project
    registrations = registration_records.read_all(
        filter={'project_id': project_id}
    )

    # Retrieve specific experiment 
    retrieved_expt = expt_records.read(project_id=project_id, expt_id=expt_id)
    retrieved_expt.pop('relations')

    # Create an optimisation run under specified experiment for current project
    #optim_run_id = optim_run_template.safe_substitute({'id': uuid.uuid1().hex})
    optim_run_id = optim_run_template.safe_substitute({'id': nni.get_trial_id()})
    new_optim_run = run_records.create(
        project_id=project_id,
        expt_id=expt_id,
        run_id=optim_run_id,
        details=params
    )

    keys={
        'project_id': project_id, 
        'expt_id': expt_id, 
        'run_id': optim_run_id
    }

    # Train on experiment-run combination
    results = start_expt_run_training(
        keys=keys,
        registrations=registrations,
        experiment=retrieved_expt,
        run=new_optim_run,
        dockerised=dockerised,
        log_msgs=log_msgs,
        verbose=verbose
    )

    # Archive results in database
    model_records.create(
        project_id=project_id,
        expt_id=expt_id,
        run_id=optim_run_id,
        details=results
    )

    # Calculate validation statistics for experiment-run combination
    participants = [record['participant']['id'] for record in registrations]
    validation_stats = start_expt_run_inference(
        keys=keys,
        participants=participants,
        registrations=registrations,
        experiment=retrieved_expt,
        run=new_optim_run,
        metas=['evaluate'],
        dockerised=dockerised,
        log_msgs=log_msgs,
        verbose=verbose,
        version=None # defaults to final state of federated grid
    )

    grouped_statistics = {}
    for participant_id, inference_stats in validation_stats.items():

        # Store output metadata into database
        worker_key = (participant_id, project_id, expt_id, optim_run_id)
        validation_records.create(*worker_key, details=inference_stats)

        # Culminate into collection of metrics
        for metric_opt in ['accuracy', 'roc_auc_score', 'pr_auc_score', 'f_score']:
            metric_collection = grouped_statistics.get(metric_opt, [])
            metric_collection.append(inference_stats[metric_opt])
            grouped_statistics[metric_opt] = metric_collection

    # Log all statistics to MLFlow
    mlf_logger.log(
        accumulations={(project_id, expt_id, optim_run_id): validation_stats}
    )

    # Calculate average of all statistics as benchmarks for model performance
    avg_statistics = {
        metric: (sum(metric_collection)/len(metric_collection))
        for metric, metric_collection in grouped_statistics.items()
    }

    # Indicate the target metric to be used for optimisation
    target_metric = avg_statistics.pop(metric)
    avg_statistics['default'] = target_metric

    # Log to NNI
    nni.report_final_result(avg_statistics)
    logging.debug(f"{project_id}_{expt_id}_{optim_run_id} - Average validation statistics: {avg_statistics}")


def get_params():
    parser = argparse.ArgumentParser(
        description="Run a Federated Learning experiment"
    )

    parser.add_argument(
        "--project_id",
        "-pid",
        type=str,
        required=True,
        help="ID of project which experimental model is registered under"
    )

    parser.add_argument(
        "--expt_id",
        "-eid",
        type=str,
        required=True,
        help="ID of experiment model to be tested"
    )

    # Possible metrics -> [accuracy, roc_auc_score, pr_auc_score, f_score]
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        required=True,
        help="Default metric to be PRIORITISED for optimisation"
    )

    parser.add_argument(
        "--dockerised",
        "-d",
        action="store_true",
        help="if set, federated cycle will use dockerised port orchestrations"
    )

    parser.add_argument(
        "--log_msgs",
        "-l",
        action="store_true",
        help="if set, websocket workers will log all intermediary operations"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, logging will be started in verbose mode"
    )

    # parser.add_argument(
    #     "--kwargs",
    #     "-k",
    #     nargs=2,
    #     action="append"
    # )

    args, _ = parser.parse_known_args()
    return args

##########
# Script #
##########

if __name__ == "__main__":

    try:
        # Get parameters from tuner defined in NNI
        tuner_params = nni.get_next_parameter()
        logging.debug(f"Detected hyperparameter set: {tuner_params}")

        # params = vars(merge_parameter(get_params(), tuner_params))
        params = {**vars(get_params()), **tuner_params}
        main(**params)
        
    except Exception as e:
        logging.error(f"Erred while tuning! Error: {e}")
        raise
    

