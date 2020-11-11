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
from typing import Tuple, Dict

# Libs
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker
# from syft.workers.node_client import NodeClient

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

# Synergos & HardwareStats logging
from SynergosLogger.init_logging import logging
from SynergosLogger import syn_logger_config
from HardwareStatsLogger import Sysmetrics

##################
# Configurations #
##################

out_dir = app.config['OUT_DIR']

db_path = app.config['DB_PATH']
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

# Hardwarestats logging
HARDWARE_STATS_LOGGER = syn_logger_config.SYSMETRICS['HARDWARE_STATS_LOGGER']
file_path = os.path.abspath(__file__)


#############
# Functions #
#############

def enumerate_expt_run_conbinations(
    experiments: list,
    runs: list,
    registrations: list,
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
        registrations (list): Registry of all participants involved
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
                    'auto_align': auto_align,
                    'dockerised': dockerised, 
                    'log_msgs': log_msgs, 
                    'verbose': verbose
                }
                combinations[combination_key] = project_expt_run_params

    return combinations


def start_expt_run_inference(
    keys: dict, 
    action: str,
    participants: list, 
    registrations: list, 
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
        participants (str): Specified IDs of participants requesting predictions
        registrations (list): Registry of all participants involved
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
    Returns:
        Statistics (dict)
    """

    # Start hardware logging process for training
    hw_logging_process = Sysmetrics.run(hardware_stats_logger=HARDWARE_STATS_LOGGER, 
                                        file_path=file_path, class_name="", 
                                        function_name="start_expt_run_inference")  

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

        # Close WSCW local objects once training process is completed (if possible)
        # (i.e. graceful termination)
        terminate_connections(ttp=ttp, workers=workers)

        return participants_inferences

    # logging.info(f"Current combination: {keys}")
    logging.info(f"Current combination", description=keys)

    # Send initialisation signal to all remote worker WSSW objects
    governor = Governor(auto_align=auto_align, dockerised=dockerised, **keys)
    governor.initialise(reg_records=registrations)

    participants_inferences = infer_combination()
    #logging.debug(f"Aggregated predictions: {participants_inferences}")

    # Stats will only be computed for relevant participants
    # (i.e. contributed datasets used for inference)
    relevant_participants = list(participants_inferences.keys())
    relevant_registrations = [
        reg_records 
        for reg_records in registrations
        if reg_records['participant']['id'] in relevant_participants
    ]

    # Convert collection of object IDs accumulated from minibatch 
    analyser = Analyser(**keys, inferences=participants_inferences, metas=metas)
    polled_stats = analyser.infer(reg_records=relevant_registrations)
    # logging.debug(f"Polled statistics: {polled_stats}")
    logging.debug(f"Polled statistics", description=polled_stats)

    # Send terminate signal to all participants' worker nodes
    # governor = Governor(dockerised=dockerised, **keys)
    governor.terminate(reg_records=registrations)

    # Terminate the hardware logging process once training has completed
    Sysmetrics.terminate(hw_logging_process)
    
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
        action = kwargs.pop('action')
        participants = kwargs.pop('participants')
        metas = kwargs.pop('metas')
        version = kwargs.pop('version')
        project_combinations = enumerate_expt_run_conbinations(**kwargs)

        for _, combination in project_combinations.items(): 
            combination.update({
                'action': action,
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