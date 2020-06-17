#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import time
from glob import glob
from pathlib import Path

# Libs
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.node_client import NodeClient

# Custom
from rest_rpc import app
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.model import Model
from rest_rpc.training.core.federated_learning import FederatedLearning
from rest_rpc.training.core.utils import Governor, RPCFormatter, ModelRecords
from rest_rpc.training.core.server import (
    connect_to_ttp, 
    connect_to_workers,
    load_selected_experiment,
    load_selected_run,
    terminate_connections
)
from rest_rpc.evaluation.core.utils import Analyser

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

#############
# Functions #
#############

def enumerate_expt_run_conbinations(
    experiments: list,
    runs: list,
    registrations: list,
    dockerised: bool = True,
    log_msgs: bool = True,
    verbose: bool = True,
) -> dict:
    """ Enumerates all registered combinations of experiment models and run
        configurations for a SINGLE project in preparation for bulk operations.

    Args:
        experiments (list): All experimental models to be reconstructed
        runs (dict): All hyperparameter sets to be used during grid FL inference
        registrations (list): Registry of all participants involved
        dockerised (bool): Toggles if current FL grid is containerised or not. 
            If true (default), hosts & ports of all participants are locked at
            "0.0.0.0" & 8020 respectively. Otherwise, participant specified
            configurations will be used (grid architecture has to be finalised).
        log_msgs (bool): Toggles if messages are to be logged
        verbose (bool): Toggles verbosity of logs for WSCW objects
    Returns:
        Combinations (dict)
    """
    combinations = {}
    for expt_record in experiments:
        curr_expt_id = expt_record['key']['expt_id']

        for run_record in runs:
            run_key = run_record['key']
            r_project_id = run_key['project_id']
            r_expt_id = run_key['expt_id']
            r_run_id = run_key['run_id']

            if r_expt_id == curr_expt_id:

                combination_key = (r_project_id, r_expt_id, r_run_id)
                project_expt_run_params = {
                    'keys': run_key,
                    'registrations': registrations,
                    'experiment': expt_record,
                    'run': run_record,
                    'dockerised': dockerised, 
                    'log_msgs': log_msgs, 
                    'verbose': verbose
                }
                combinations[combination_key] = project_expt_run_params

    return combinations


def start_expt_run_inference(
    keys: dict, 
    participants: list, 
    registrations: list, 
    experiment: dict, 
    run: dict, 
    metas: list = ['train', 'evaluate', 'predict'],
    dockerised: bool = True, 
    log_msgs: bool = True, 
    verbose: bool = True
) -> dict:
    """ Trains a model corresponding to a SINGLE experiment-run combination

    Args:
        keys (dict): Relevant Project ID, Expt ID & Run ID
        participants (str): Specified IDs of participants requesting predictions
        registrations (list): Registry of all participants involved
        experiment (dict): Parameters for reconstructing experimental model
        run (dict): Hyperparameters to be used during grid FL inference
        metas (list): Type(s) of datasets to perform inference on
        dockerised (bool): Toggles if current FL grid is containerised or not. 
            If true (default), hosts & ports of all participants are locked at
            "0.0.0.0" & 8020 respectively. Otherwise, participant specified
            configurations will be used (grid architecture has to be finalised).
        log_msgs (bool): Toggles if messages are to be logged
        verbose (bool): Toggles verbosity of logs for WSCW objects
    Returns:
        Statistics (dict)
    """

    def infer_combination():

        # Create worker representation for local machine as TTP
        ttp = connect_to_ttp(log_msgs=log_msgs, verbose=verbose)

        # Complete WS handshake with participants
        workers = connect_to_workers(
            keys=keys,
            reg_records=registrations,
            dockerised=dockerised,
            log_msgs=log_msgs,
            verbose=verbose
        )

        # Restore model from archive (must differentiate between Normal & SNN)
        model = load_selected_experiment(expt_record=experiment)
        archived_weight_paths = model_records.read(**keys)

        # Check if a trained model exists
        if archived_weight_paths:

            archived_global_weights = th.load(archived_weight_paths['global']['path'])
            model.load_state_dict(archived_global_weights)

            args = load_selected_run(run_record=run)
        
            #############################################
            # Inference V1: Assume TTP's role is robust #
            #############################################

            fl_expt = FederatedLearning(args, ttp, workers, model)
            fl_expt.load(shuffle=False) # for re-assembly during inference
            
            # Only infer for specified participant on his/her own test dataset
            participants_inferences = fl_expt.evaluate(
                metas=metas,
                workers=participants
            )

        else:
            # No trained model --> No available results
            participants_inferences = {
                participant: {} 
                for participant in participants
            }

        # Close WSCW local objects once training process is completed (if possible)
        # (i.e. graceful termination)
        terminate_connections(ttp=ttp, workers=workers)

        return participants_inferences

    logging.info(f"Current combination: {keys}")

    # Send initialisation signal to all remote worker WSSW objects
    governor = Governor(dockerised=dockerised, **keys)
    governor.initialise(reg_records=registrations)

    participants_inferences = infer_combination()
    logging.debug(f"Aggregated predictions: {participants_inferences}")

    # Convert collection of object IDs accumulated from minibatch 
    analyser = Analyser(**keys, inferences=participants_inferences, metas=metas)
    polled_stats = analyser.infer(reg_records=registrations)
    logging.debug(f"Polled statistics: {polled_stats}")

    # Send terminate signal to all participants' worker nodes
    governor = Governor(dockerised=dockerised, **keys)
    governor.terminate(reg_records=registrations)

    return polled_stats


def start_proc(multi_kwargs: dict) -> dict:
    """ Automates the inference of Federated models of different architectures
        and parameter sets

    Args:
        multi_kwargs (dict): Experiments & models to be tested
    Returns:
        Statistics of each specified project-expt-run configuration (dict)
    """
    all_statistics = {}
    for _, kwargs in multi_kwargs.items():

        participants = kwargs.pop('participants')
        metas = kwargs.pop('metas')
        project_combinations = enumerate_expt_run_conbinations(**kwargs)

        for _, combination in project_combinations.items(): 
            combination.update({
                'participants': participants, 
                "metas": metas
            })

        completed_project_inferences = {
            combination_key: start_expt_run_inference(**kwargs) 
            for combination_key, kwargs in project_combinations.items()
        }

        all_statistics.update(completed_project_inferences)

    return all_statistics