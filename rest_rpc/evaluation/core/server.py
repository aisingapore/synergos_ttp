#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import time
from glob import glob
from logging import NOTSET
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Libs
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker
# from syft.workers.node_client import NodeClient

# Custom
from rest_rpc import app
from rest_rpc.training.core.federated_learning import FederatedLearning
from rest_rpc.training.core.utils import Governor, RPCFormatter
from rest_rpc.training.core.server import (
    connect_to_ttp, 
    connect_to_workers,
    load_selected_experiment,
    load_selected_run,
    terminate_connections
)
from rest_rpc.evaluation.core.utils import Analyser
from synarchive.training import ModelRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/core/server.py logged", Description="No Changes")

#############
# Functions #
#############

def enumerate_expt_run_conbinations(
    experiments: list,
    runs: list,
    auto_align: bool = True,
    dockerised: bool = True,
    log_msgs: bool = True,
    verbose: bool = True,
) -> dict:
    """ Enumerates all registered combinations of experiment models and run
        configurations for a SINGLE project in preparation for bulk operations.

    Args:
        experiments (list): All experimental models to be reconstructed
        runs (dict): All hyperparameter sets to be used during grid FL inference
        auto_align (bool): Toggles if multiple feature alignments will be used
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
            collab_id = run_key['collab_id']
            project_id = run_key['project_id']
            expt_id = run_key['expt_id']
            run_id = run_key['run_id']

            if expt_id == curr_expt_id:

                combination_key = (collab_id, project_id, expt_id, run_id)
                collab_project_expt_run_params = {
                    'keys': run_key,
                    'experiment': expt_record,
                    'run': run_record,
                    'auto_align': auto_align,
                    'dockerised': dockerised, 
                    'log_msgs': log_msgs, 
                    'verbose': verbose
                }
                combinations[combination_key] = collab_project_expt_run_params

    return combinations


def start_expt_run_inference(
    keys: dict, 
    action: str,
    grid: List[Dict[str, Any]],
    participants: list, 
    experiment: dict, 
    run: dict, 
    metas: list = ['train', 'evaluate', 'predict'],
    auto_align: bool = True,
    dockerised: bool = True, 
    log_msgs: bool = True, 
    verbose: bool = True,
    version: Tuple[str, str] = None
) -> dict:
    """ Trains a model corresponding to a SINGLE experiment-run combination

    Args:
        keys (dict): Relevant Project ID, Expt ID & Run ID
        action (str): Type of machine learning operation to be executed
        grid (list(dict))): Registry of participants' node information
        participants (str): Specified IDs of participants requesting predictions
        experiment (dict): Parameters for reconstructing experimental model
        run (dict): Hyperparameters to be used during grid FL inference
        metas (list): Type(s) of datasets to perform inference on
        auto_align (bool): Toggles if multiple feature alignments will be used
        dockerised (bool): Toggles if current FL grid is containerised or not. 
            If true (default), hosts & ports of all participants are locked at
            "0.0.0.0" & 8020 respectively. Otherwise, participant specified
            configurations will be used (grid architecture has to be finalised).
        log_msgs (bool): Toggles if messages are to be logged
        verbose (bool): Toggles verbosity of logs for WSCW objects
        version (tuple): Specifies set of weights from (round, epoch) to load
    Returns:
        Statistics (dict)
    """

    def infer_combination():

        logging.warn(f"---> Participants declared: {participants}")
        # Create worker representation for local machine as TTP
        ttp = connect_to_ttp(log_msgs=log_msgs, verbose=verbose)

        # Complete WS handshake with participants
        workers = connect_to_workers(
            grid=grid,
            log_msgs=log_msgs,
            verbose=verbose
        )

        # Instantiate grid environment
        model = load_selected_experiment(expt_record=experiment)
        args = load_selected_run(run_record=run)

        # Check if current expt-run combination has already been trained
        combination_archive = model_records.read(**keys)
        if combination_archive:

            stripped_archive = rpc_formatter.strip_keys(
                combination_archive, 
                concise=True
            )

            # Restore models from archive (differentiated between Normal & SNN)
            fl_expt = FederatedLearning(
                action=action,
                arguments=args, 
                crypto_provider=ttp, 
                workers=workers, 
                reference=model
            )
            fl_expt.load(
                archive=stripped_archive,
                shuffle=False,   # for re-assembly during inference
                version=version
            ) 
  
            # Only infer for specified participant on his/her own test dataset
            participants_inferences, _ = fl_expt.evaluate(
                metas=metas,
                workers=participants
            )

        else:
            # No trained model --> No available results
            participants_inferences = {
                participant: {} 
                for participant in participants
            }

        logging.warn(f"participant_interference: {participants_inferences}")

        # Close WSCW local objects once training process is completed (if possible)
        # (i.e. graceful termination)
        terminate_connections(ttp=ttp, workers=workers)

        return participants_inferences

    logging.info(
        f"Evaluation - Federated combination tracked.",
        keys=keys,
        ID_path=SOURCE_FILE,
        ID_function=start_expt_run_inference.__name__
    )

    # Send initialisation signal to all remote worker WSSW objects
    governor = Governor(auto_align=auto_align, dockerised=dockerised, **keys)
    governor.initialise(grid=grid)

    participants_inferences = infer_combination()

    logging.log(
        level=NOTSET,
        event=f"Evaluation - Aggregated predictions tracked.",
        participants_inferences=participants_inferences,
        ID_path=SOURCE_FILE,
        ID_function=start_expt_run_inference.__name__
    )

    # Stats will only be computed for relevant participants
    # (i.e. contributed datasets used for inference)
    analyser = Analyser(
        auto_align=auto_align, 
        inferences=participants_inferences, 
        metas=metas,
        **keys
    )
    polled_stats = analyser.infer(grid=grid)
    logging.warn(f"---> polled stats: {polled_stats}")

    logging.debug(
        f"Evaluation - Polled statistics tracked.", 
        polled_stats=polled_stats,
        ID_path=SOURCE_FILE,
        ID_function=start_expt_run_inference.__name__
    )

    # Send terminate signal to all participants' worker nodes
    governor.terminate(grid=grid)
    
    return polled_stats


def evaluate_proc(grid: List[Dict[str, Any]], multi_kwargs: dict) -> dict:
    """ Automates the inference of Federated models of different architectures
        and parameter sets

    Args:
        grid (list(dict))): Registry of participants' node information
        multi_kwargs (dict): Experiments & models to be tested
    Returns:
        Statistics of each specified project-expt-run configuration (dict)
    """
    all_statistics = {}
    for _, kwargs in multi_kwargs.items():
        action = kwargs.pop('action')
        participants = kwargs.pop('participants')
        metas = kwargs.pop('metas')
        version = kwargs.pop('version')
        project_combinations = enumerate_expt_run_conbinations(**kwargs)

        for _, combination in project_combinations.items(): 
            combination.update({
                'action': action,
                'grid': grid,
                'participants': participants, 
                'metas': metas,
                'version': version
            })

        completed_project_inferences = {
            combination_key: start_expt_run_inference(**kwargs) 
            for combination_key, kwargs in project_combinations.items()
        }

        all_statistics.update(completed_project_inferences)

    return all_statistics