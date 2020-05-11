#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import argparse
import asyncio
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
from rest_rpc import app
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.model import Model
from rest_rpc.training.core.federated_learning import FederatedLearning
from rest_rpc.training.core.utils import Governor, RPCFormatter
##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

out_dir = app.config['OUT_DIR']

rpc_formatter = RPCFormatter()

# Instantiate a local hook for coordinating clients
grid_hook = sy.TorchHook(th)

#############
# Functions #
#############

def connect_to_ttp(log_msgs=False, verbose=False):
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
    """
    # Create a virtual worker representing the TTP
    ttp = sy.VirtualWorker(
        hook=grid_hook, 
        id='ttp',
        is_client_worker=False,
        log_msgs=log_msgs,
        verbose=verbose    
    )

    # Replace point of reference within federated hook with TTP
    #grid_hook.local_worker = ttp
    #assert (ttp is grid_hook.local_worker)

    logging.debug(f"Local worker w.r.t grid hook: {grid_hook.local_worker}")
    logging.debug(f"Local worker w.r.t env      : {sy.local_worker}")
    """
    return grid_hook.local_worker#sy.local_worker#ttp


def connect_to_workers(reg_records, verbose=False):
    """ Create client workers for participants to their complete WS connections

    Args:
        reg_records (list(tinydb.database.Document))): Registry of participants
    Returns:
        workers (list(WebsocketClientWorker))
    """
    workers = []
    for reg_record in reg_records:

        # Remove redundant fields
        config = rpc_formatter.strip_keys(reg_record['participant'].copy())
        config.pop('f_port')

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


def load_selected_runs(fl_params):
    """ Load in specified federated experimental parameters to be conducted from
        configuration files

    Args:
        fl_params (dict): Experiment Ids of experiments to be run
    Returns:
        runs (dict(str,Arguments))
    """
    runs = {run_id: Arguments(**params) for run_id, params in fl_params.items()}

    logging.debug(f"Runs loaded: {runs.keys()}")

    return runs


def load_selected_models(model_params):
    """ Load in specified federated model architectures to be used for training
        from configuration files

    Args:
        model_params (dict): Specified models to be trained
    Returns:
        models (dict(str,Model))
    """
    models = {name: Model(structure) for name, structure in model_params.items()}

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
    for worker in workers:

        # Superclass of websocketclient (baseworker) contains a worker registry 
        # which caches the websocketclient objects when the auto_add variable is
        # set to True. This registry is indexed by the websocketclient ID and 
        # thus, recreation of the websocketclient object will not get replaced 
        # in the registry if the ID is the same. If obj is not removed from
        # local registry before the WS connection is closed, this will cause a 
        # `websocket._exceptions.WebSocketConnectionClosedException: socket is 
        # already closed.` error since it is still referencing to the previous 
        # websocketclient connection that was closed. The solution 
        # was to simply call the remove_worker_from_local_worker_registry() 
        # method on the websocketclient object before closing its connection.
        worker.remove_worker_from_local_worker_registry()

        # Now close the websocket connection
        worker.close() 


def start_proc(kwargs):
    """ Automates the execution of Federated learning experiments on different
        hyperparameter sets & model architectures

    Args:
        kwargs (dict): Experiments & models to be tested
    Returns:
        Path-to-trained-models (list(str))
    """
    # Extract intialisation parameters
    log_msgs = kwargs['log_msgs']
    is_verbose = kwargs['verbose']
    keys = kwargs['keys']
    is_dockerised = kwargs['dockerised']
    reg_records = kwargs['registrations']
    
    logging.info([rr['participant'] for rr in reg_records])

    # Create worker representation for local machine as TTP
    ttp = connect_to_ttp(log_msgs=log_msgs, verbose=is_verbose)

    # Initialise all remote worker WSSW objects
    governor = Governor(dockerised=is_dockerised, **keys)
    governor.initialise(reg_records=reg_records)

    # Complete WS handshake with participants
    workers = connect_to_workers(reg_records, verbose=is_verbose)

    logging.info(f"Workers: {workers}")

    # Load in all selected runs
    run_records = kwargs['runs']
    runs = load_selected_runs(run_records)

    # Load in all selected experiment models
    expt_records = kwargs['experiments']
    selected_models = load_selected_models(expt_records)

    completed_trainings = {}
    # Test out each sepcified model
    for expt_id, model in selected_models.items():

        # Test out each specified run
        for run_id, args in runs.items():
            logging.info(f"Run ID: {run_id}, Arguments: {vars(args)}")

            # Perform a Federated Learning experiment
            fl_expt = FederatedLearning(args, ttp, workers, model)
            fl_expt.load()
            fl_expt.fit()

            # Export trained model weights/biases for persistence
            res_dir = os.path.join(out_dir, keys['project_id'], expt_id, run_id)
            Path(res_dir).mkdir(parents=True, exist_ok=True)

            out_paths = fl_expt.export(res_dir)

            logging.info(f"Final model: {fl_expt.global_model.state_dict()}")
            logging.info(f"Final model stored at {out_paths}")
            logging.info(f"Loss history: {fl_expt.loss_history}")

            completed_trainings[(expt_id, run_id)] = out_paths

    # Remember to close WSCW local objects once the training process is completed
    # (i.e. graceful termination)
    terminate_connections(workers)

    # Finally, terminate WSSW remote objects for all participants' worker nodes
    # (if no other project is running)
    governor.terminate(reg_records=reg_records)

    return completed_trainings

##########
# Script #
##########

if __name__ == "__main__":
    
    """
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
    """

