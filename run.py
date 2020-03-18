#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import logging
import os
import time
from glob import glob
from multiprocessing import Process
from pathlib import Path

# Libs
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from arguments import Arguments
from config import SRC_DIR, server_params, model_params, fl_params
from model import Model
from federated_learning import FederatedLearning

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Assumptions: All participant already have their WS server workers initialised
# Instantiate a local hook for coordinating clients
grid_hook = sy.TorchHook(th)

#############
# Functions #
#############

def connect_to_ttp(verbose=False):
    """ Creates coordinating TTP on the local machine, and makes it the point of
        reference for subsequent federated operations. Since this implementation
        is that of the master-slave paradigm, the local machine would be
        considered as the subject node in the network, allowing the TTP to be
        represented by a VirtualWorker.

        (While it is possible for TTP to be a WebsocketClientWorker, it would 
        make little sense to route messages via a WS connection to itself, since
        that would add unnecessary network overhead.)
    
    Returns:
        ttp (sy.VirtualWorker)
    """
    # Create a virtual worker representing the TTP
    ttp = sy.VirtualWorker(
        hook=grid_hook, 
        id='ttp',
        is_client_worker=False,
        log_msgs=True,
        verbose=verbose    
    )

    # Replace point of reference within federated hook with TTP
    grid_hook.local_worker = ttp
    assert (ttp is grid_hook.local_worker)

    logging.debug(f"Local worker w.r.t hook: {grid_hook.local_worker}")
    logging.debug(f"Local worker w.r.t env : {sy.local_worker}")

    return ttp


def connect_to_workers(verbose=False):
    """ Create client workers for participants to their complete WS connections

    Returns:
        workers (list(WebsocketClientWorker))
    """
    workers = []
    for worker_idx, config in server_params.items():

        # Replace verbosity settings with local verbosity settings
        config['verbose'] = verbose
        
        curr_worker = WebsocketClientWorker(
            hook=grid_hook,
            is_client_worker=True,
            **config
        )
        workers.append(curr_worker)

    logging.debug(f"Participants: {[w.id for w in workers]}")

    return workers


def load_selected_experiments(experiment_ids):
    """ Load in specified federated experimental parameters to be conducted from
        configuration files

    Args:
        experimental_ids (list(str)): Experiment Ids of experiments to be run
    Returns:
        experiments (list(Arguments))
    """
    experiments = {}
    for exp_id, params in fl_params.items():

        # If current model is selected for training, load it
        if exp_id in experiment_ids:
            experiments[exp_id] = Arguments(**params)

    logging.debug(f"Experiments loaded: {experiments.keys()}")

    return experiments


def load_selected_models(model_names):
    """ Load in specified federated model architectures to be used for training
        from configuration files

    Args:
        model_names (list(str)): Specified models to be trained
    Returns:
        models (list(Model))
    """
    models = {}
    for name, structure in model_params.items():

        # If current model is selected for training, load it
        if name in model_names:
            models[name] = Model(structure)

    logging.debug(f"Models loaded: {models.keys()}")

    return models


def terminate_connections(workers):
    """ Terminates the WS connections between remote WebsocketServerWorkers &
        local WebsocketClientWorkers

    Args:
        workers (list(WebsocketClientWorkers)): Participants of the FL training
    Returns:
        closed workers (list(WebsocketClientWorkers))
    """
    
    workers_closed = [w.close() for w in workers]

    logging.debug(f"Workers closed: {workers_closed}")

    return workers_closed


def start_proc(kwargs):
    """ Automates the execution of Federated learning experiments on different
        hyperparameter sets & model architectures

    Args:
        kwargs (dict): Experiments & models to be tested
    Returns:
        Path-to-trained-models (list(str))
    """
    # Create worker representation for local machine as TTP
    is_verbose = kwargs['verbose']
    ttp = connect_to_ttp(is_verbose)
    
    # Complete WS handshake with participants
    workers = connect_to_workers()

    # Load in all selected experiments
    experiment_ids = kwargs['experiments']
    experiments = load_selected_experiments(experiment_ids)

    # Load in all selected models
    model_names = kwargs['models']
    selected_models = load_selected_models(model_names)

    trained_models = []
    # Test out each specified experiment
    for exp_id, args in experiments.items():

        # Test out each sepcified model
        for name, model in selected_models.items():

            # Perform a Federated Learning experiment
            fl_expt = FederatedLearning(args, ttp, workers, model)
            fl_expt.load()
            fl_expt.fit()

            # Export trained model weights/biases for persistence
            out_dir = os.path.join(SRC_DIR, "outputs", exp_id, name)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            out_paths = fl_expt.export(out_dir)

            logging.info(f"Final model: {fl_expt.global_model.state_dict()}")
            logging.info(f"Final model stored at {out_paths}")
            logging.info(f"Loss history: {fl_expt.loss_history}")

            trained_models.append(out_paths)

    # Remember to close workers once the training process is completed
    # (i.e. graceful termination)
    #terminate_connections(workers)
    
    return trained_models

##########
# Script #
##########

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run a Federated Learning experiment."
    )

    parser.add_argument(
        "--models",
        "-m",
        type=str, 
        nargs="+",
        required=True,
        help="Model architecture to load"
    )

    parser.add_argument(
        "--experiments",
        "-e",
        type=str, 
        nargs="+",
        required=True,
        help="Port number of the websocket server worker, e.g. --port 8020"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client worker will be started in verbose mode"
    )

    kwargs = vars(parser.parse_args())
    logging.debug(f"TTP Parameters: {kwargs}")

    start_proc(kwargs)

##############
# Deprecated #
##############
"""
bob = WebsocketClientWorker(
    host='federated-learning-staging-vm-2.eastus.cloudapp.azure.com',
    hook=hook,
    id="Bob", 
    port=8020,
    is_client_worker=True,
    log_msgs=True,
    verbose=True
)
"""
