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
from rest_rpc.training.core.utils import Governor, RPCFormatter
from rest_rpc.training.core.server import (
    connect_to_ttp, 
    connect_to_workers,
    load_selected_experiment,
    load_selected_run,
    terminate_connections
)
from rest_rpc.evaluation.core.utils import Analyser
from synalgo import FederatedLearning
from synarchive.training import ModelRecords

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

db_path = app.config['DB_PATH']
model_records = ModelRecords(db_path=db_path)

rpc_formatter = RPCFormatter()

logging = app.config['NODE_LOGGER'].synlog
logging.debug("evaluation/core/server.py logged", Description="No Changes")

#############
# Functions #
#############

def execute_combination_inference(
    grid: List[Dict[str, Any]],
    keys: dict, 
    action: str,
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
        ID_function=execute_combination_inference.__name__
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
        ID_function=execute_combination_inference.__name__
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
        ID_function=execute_combination_inference.__name__
    )

    # Send terminate signal to all participants' worker nodes
    governor.terminate(grid=grid)
    
    return polled_stats
