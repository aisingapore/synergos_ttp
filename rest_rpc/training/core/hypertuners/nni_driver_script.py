#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import argparse
import logging
import os
from string import Template
from typing import Dict, List, Any

# Libs
import nni
from nni.utils import merge_parameter

# Custom
from rest_rpc import app
from rest_rpc.training.core.server import start_expt_run_training
from rest_rpc.training.core.utils import RPCFormatter
from rest_rpc.evaluation.core.server import start_expt_run_inference
from rest_rpc.evaluation.core.utils import MLFlogger
from synarchive.connection import (
    ProjectRecords,
    ExperimentRecords,
    RunRecords,
    RegistrationRecords
)
from synarchive.training import ModelRecords
from synarchive.evaluation import ValidationRecords

##################
# Configurations #
##################

grid_idx = app.config['GRID']

db_path = app.config['DB_PATH']
project_records = ProjectRecords(db_path=db_path)
expt_records = ExperimentRecords(db_path=db_path)
run_records = RunRecords(db_path=db_path)
registration_records = RegistrationRecords(db_path=db_path)
model_records = ModelRecords(db_path=db_path)
validation_records = ValidationRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

mlflow_dir = app.config['MLFLOW_DIR']
mlf_logger = MLFlogger()

# Template for generating optimisation run ID
optim_prefix = "optim_run_"
optim_run_template = Template(optim_prefix + "$id")

#############
# Functions #
#############

def main(
    collab_id: str,
    project_id: str,
    expt_id: str,
    metric: str,
    auto_align: bool = True,
    dockerised: bool = True, 
    log_msgs: bool = True, 
    verbose: bool = True,
    **params
):
    """ Stores run parameters, train model on specified parameter set, and
        extract validation statistics on validation sets across the federated
        grid.
        Note:
        This function is ALWAYS executed from a local point of reference
        (i.e. TTP not Director). This means that a consumable grid already
        exists and is already pre-allocated.

    Args:
        collab_id (str): Collaboration ID of current collaboration
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
        filter={'collab_id': collab_id, 'project_id': project_id}
    )

    # Consume a grid for running current federated combination
    all_grids = rpc_formatter.extract_grids(registrations)
    allocated_grid = all_grids[grid_idx]

    # Retrieve specific project
    retrieved_project = project_records.read(
        collab_id=collab_id,
        project_id=project_id
    )
    project_action = retrieved_project['action']

    # Retrieve specific experiment 
    retrieved_expt = expt_records.read(
        collab_id=collab_id,
        project_id=project_id, 
        expt_id=expt_id
    )

    # Create an optimisation run under specified experiment for current project
    optim_run_id = optim_run_template.safe_substitute({'id': nni.get_trial_id()})
    
    keys = {
        'collab_id': collab_id,
        'project_id': project_id, 
        'expt_id': expt_id, 
        'run_id': optim_run_id
    }
    
    run_records.create(**keys, details=params)
    new_optim_run = run_records.read(**keys)

    # Train on experiment-run combination
    results = start_expt_run_training(
        keys=keys,
        action=project_action,
        grid=allocated_grid,
        experiment=retrieved_expt,
        run=new_optim_run,
        auto_align=auto_align,
        dockerised=dockerised,
        log_msgs=log_msgs,
        verbose=verbose
    )

    # Archive results in database
    model_records.create(**keys, details=results)

    # Calculate validation statistics for experiment-run combination
    participants = [record['participant']['id'] for record in registrations]
    validation_stats = start_expt_run_inference(
        keys=keys,
        action=project_action,
        grid=allocated_grid,
        participants=participants,
        experiment=retrieved_expt,
        run=new_optim_run,
        metas=['evaluate'],
        auto_align=auto_align,
        dockerised=dockerised,
        log_msgs=log_msgs,
        verbose=verbose,
        version=None # defaults to final state of federated grid
    )

    combination_key = (collab_id, project_id, expt_id, optim_run_id)

    grouped_statistics = {}
    for participant_id, inference_stats in validation_stats.items():

        # Store output metadata into database
        worker_key = (participant_id,) + combination_key
        validation_records.create(*worker_key, details=inference_stats)

        # Culminate into collection of metrics
        supported_metrics = ['accuracy', 'roc_auc_score', 'pr_auc_score', 'f_score']
        for metric_opt in supported_metrics:

            metric_collection = grouped_statistics.get(metric_opt, [])
            curr_metrics = inference_stats['evaluate']['statistics'][metric_opt]
            metric_collection.append(curr_metrics)
            grouped_statistics[metric_opt] = metric_collection

    # Log all statistics to MLFlow
    mlf_logger.log(accumulations={combination_key: validation_stats})

    # Calculate average of all statistics as benchmarks for model performance
    calculate_avg_stats = lambda x: (sum(x)/len(x))
    avg_statistics = {
        metric: calculate_avg_stats([
            calculate_avg_stats(p_metrics) 
            for p_metrics in metric_collection
        ])
        for metric, metric_collection in grouped_statistics.items()
    }

    # Indicate the target metric to be used for optimisation
    target_metric = avg_statistics.pop(metric)
    avg_statistics['default'] = target_metric

    # Log to NNI
    nni.report_final_result(avg_statistics)
    logging.debug(f"{collab_id}_{project_id}_{expt_id}_{optim_run_id} - Average validation statistics: {avg_statistics}")


def get_params():
    parser = argparse.ArgumentParser(
        description="Run a Federated Learning experiment"
    )

    parser.add_argument(
        "--collab_id",
        "-cid",
        type=str,
        required=True,
        help="ID of project which experimental model is registered under"
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

    args, _ = parser.parse_known_args()
    return args

##########
# Script #
##########

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

    # try:
    # Get parameters from tuner defined in NNI
    tuner_params = nni.get_next_parameter()
    logging.debug(f"Detected hyperparameter set: {tuner_params}")

    # params = vars(merge_parameter(get_params(), tuner_params))
    params = {**vars(get_params()), **tuner_params}
    main(**params)
        
    # except Exception as e:
    #     logging.error(f"Erred while tuning! Error: {e}")
    #     raise
    

