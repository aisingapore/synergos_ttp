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
from rest_rpc.evaluation.core.utils import Inferencer

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

def convert_to_stats(keys, registrations, inferences):
    """ Automates the submission of inference objects within a collection to
        their respective workers, retrieving descriptive statistics in return

    Args:
        keys (dict): Relevant Project ID, Expt ID & Run ID
        registrations (list(tinydb.database.Document))): Participant registry
        inferences (dict(worker_id, dict(str, int)))
    Returns:
        Converted statistics (dict(key, dict(worker_id, dict(str, float))))
    """
    inferencer = Inferencer(inferences=inferences, **keys)
    converted_stats = inferencer.infer(reg_records=registrations)

    return converted_stats


def start_expt_run_inference(keys: dict, participant: str, registrations: list, 
                             experiment: dict, run: dict, 
                             dockerised: bool, log_msgs: bool, verbose: bool):
    """ Trains a model corresponding to a SINGLE experiment-run combination

    Args:
        keys (dict): Relevant Project ID, Expt ID & Run ID
        ttp (sy.VirtualWorker): Allocated trusted third party in the FL grid
        workers (list(WebsocketClientWorker)): All WSCWs for each participant
        experiment (dict): Parameters for reconstructing experimental model
        run (dict): Hyperparameters to be used during grid FL training
    Returns:
        Path-to-trained-models (dict(str))
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
            fl_expt.load()
            participant_stats = fl_expt.evaluate(query=[participant])

            ####################################################################
            # Inference V2: Strictly enforce federated procedures in inference #
            ####################################################################
            # Inference is integrated into the evaluation cycle directly

            # participant_stats = fl_expt.evaluate(
            #     query=[participant],
            #     keys=keys,
            #     registrations=[
            #         record 
            #         for record in registrations
            #         if record['participant']['id'] == participant
            #     ]
            # )

        else:
            participant_stats = {participant: {}} # no trained model, no stats!

        # # Convert collection of object IDs accumulated from minibatch 
        # inferencer = Inferencer(inferences=participant_stats, **keys)
        # converted_stats = inferencer.infer(reg_records=registrations)

        # Close WSCW local objects once training process is completed (if possible)
        # (i.e. graceful termination)
        terminate_connections(ttp=ttp, workers=workers)

        # return converted_stats
        return participant_stats

    logging.info(f"Current combination: {keys}")

    # Send initialisation signal to all remote worker WSSW objects
    governor = Governor(dockerised=dockerised, **keys)
    governor.initialise(reg_records=registrations)

    results = infer_combination()

    # Send terminate signal to all participants' worker nodes
    governor = Governor(dockerised=dockerised, **keys)
    governor.terminate(reg_records=registrations)

    return results


def start_proc(multi_kwargs):
    """ Automates the inference of Federated models of different architectures
        and parameter sets

    Args:
        kwargs (dict): Experiments & models to be tested
    Returns:
        Path-to-trained-models (list(str))
    """
    all_statistics = {}
    for project_id, kwargs in multi_kwargs.items():

        participant = kwargs['participant']
        experiments = kwargs['experiments']
        runs = kwargs['runs']
        registrations = kwargs['registrations']
        is_dockerised = kwargs['dockerised']

        logging.debug(f"registrations: {registrations}")
        inference_combinations = {}
        for expt_record in experiments:
            curr_expt_id = expt_record['key']['expt_id']

            for run_record in runs:
                run_key = run_record['key']
                r_project_id = run_key['project_id']
                r_expt_id = run_key['expt_id']
                r_run_id = run_key['run_id']

                if (r_project_id == project_id) and (r_expt_id == curr_expt_id):

                    combination_key = (r_project_id, r_expt_id, r_run_id)
                    project_expt_run_params = {
                        'participant': participant,
                        'keys': run_key,
                        'registrations': registrations,
                        'experiment': expt_record,
                        'run': run_record,
                        'dockerised': is_dockerised, 
                        'log_msgs': True, 
                        'verbose': True
                    }
                    inference_combinations[combination_key] = project_expt_run_params

        logging.info(f"{inference_combinations}")

        completed_project_inferences = {
            combination_key: start_expt_run_inference(**kwargs) 
            for combination_key, kwargs in inference_combinations.items()
        }

        all_statistics.update(completed_project_inferences)

    return all_statistics