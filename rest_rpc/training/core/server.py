#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import argparse
import asyncio
import concurrent.futures
import inspect
import multiprocessing as mp
import os
import time
from logging import NOTSET
from glob import glob
from pathlib import Path

# Libs
import dill
import syft as sy
import torch as th
from pathos.multiprocessing import ProcessingPool
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from rest_rpc import app
from rest_rpc.training.core.arguments import Arguments
from rest_rpc.training.core.model import Model, ModelPlan
from rest_rpc.training.core.federated_learning import FederatedLearning
from rest_rpc.training.core.utils import Poller, Governor, RPCFormatter
from rest_rpc.training.core.custom import CustomClientWorker, CustomWSClient

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

out_dir = app.config['OUT_DIR']

cache = mp.Queue()#app.config['CACHE']

rpc_formatter = RPCFormatter()

# Instantiate a local hook for coordinating clients
# Note: `is_client=True` ensures that all objects are deleted once WS 
# connection is closed. That way, there is no need to explicitly clear objects
# in all workers, which may prematurely break PointerTensors.
grid_hook = sy.TorchHook(th)
grid_hook.local_worker.is_client_worker = False

# Configure timeout settings for WebsocketClientWorker
sy.workers.websocket_client.TIMEOUT_INTERVAL = 3600

#sy.workers.websocket_client.websocket.enableTrace(True)
REF_WORKER = sy.local_worker

"""
[Redacted - Multiprocessing]

core_count = mp.cpu_count()

# Configure dill to recursively serialise dependencies
dill.settings['recurse'] = True

[Redacted - Asynchronised FL Grid Training]
"""

# GPU customisations
gpu_count = app.config['GPU_COUNT']
gpus = app.config['GPUS']
use_gpu = app.config['USE_GPU']
device = app.config['DEVICE']

logging = app.config['NODE_LOGGER'].synlog
logging.debug("connection/core/server.py logged", Description="No Changes")

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
    # Create a virtual worker representing the TTP
    ttp = sy.VirtualWorker(
        hook=grid_hook, 
        id='ttp',
        is_client_worker=False,
        log_msgs=log_msgs,
        verbose=verbose    
    )

    # Make sure that generated TTP is self-aware! 
    # ttp.add_worker(ttp)

    # Replace point of reference within federated hook with TTP
    sy.local_worker = ttp
    grid_hook.local_worker = ttp

    logging.debug(
        "Local worker w.r.t grid hook tracked.",
        local_worker=grid_hook.local_worker,
        ID_path=SOURCE_FILE,
        ID_function=connect_to_ttp.__name__
    )
    logging.debug(
        "Local worker w.r.t env tracked.",
        local_worker=sy.local_worker,
        ID_path=SOURCE_FILE,
        ID_function=connect_to_ttp.__name__
    )
    logging.debug(
        f"Local worker's known workers tracked.",
        known_workers=grid_hook.local_worker._known_workers,
        ID_path=SOURCE_FILE,
        ID_function=connect_to_ttp.__name__
    )

    return grid_hook.local_worker


def connect_to_workers(keys, reg_records, dockerised=True, log_msgs=False, verbose=False):
    """ Create client workers for participants to their complete WS connections
        Note: At any point of time, there will always be 1 set of workers per
              main process

    Args:
        keys (dict(str)): Processing IDs for caching
        reg_records (list(tinydb.database.Document))): Registry of participants
        log_msgs (bool): Toggles if messages are to be logged
        verbose (bool): Toggles verbosity of logs for WSCW objects
    Returns:
        workers (list(WebsocketClientWorker))
    """
    workers = []
    for reg_record in reg_records:

        # Remove redundant fields
        config = rpc_formatter.strip_keys(reg_record['participant'])
        config.pop('f_port')

        # Replace log & verbosity settings with locally specified settings
        config['log_msgs'] = log_msgs
        config['verbose'] = verbose

        try:
            curr_worker = WebsocketClientWorker(
                hook=grid_hook,

                # When False, PySyft manages object clean up. Here, this is done on
                # purpose, since there is no need to propagate gradient tracking
                # tensors back to the worker node. This ensures that the grid is
                # self-contained at the TTP, and that the workers' local grid is not
                # polluted with unncessary tensors. Doing so optimizes tag searches.
                is_client_worker=False, #True

                **config
            )

        except OSError:
            curr_worker = grid_hook.local_worker._known_workers[config['id']]

        # #####################################################################
        # # Optimal Setup -  Use NodeClient objects for in-house SMPC support #
        # #####################################################################
        # # Issue: Unable to connect to WSSW object remotely, raises the 
        # #        `binascii.Error: Non-hexadecimal digit found` exception.
        # # Solution: K.I.V until issue is resolved
        # config['address'] = f"ws://{config.pop('host')}:{config.pop('port')}".strip()

        # logging.debug(f"config: {config} {type(config)}")

        # curr_worker = CustomClientWorker(
        #     hook=grid_hook,

        #     # When False, PySyft manages object clean up. Here, this is done on
        #     # purpose, since there is no need to propagate gradient tracking
        #     # tensors back to the worker node. This ensures that the grid is
        #     # self-contained at the TTP, and that the workers' local grid is not
        #     # polluted with unncessary tensors. Doing so optimizes tag searches.
        #     is_client_worker=False, #True

        #     **dict(config)
        # )

        workers.append(curr_worker)

        logging.debug(
            f"Participant's known workers tracked.",
            known_workers=curr_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=connect_to_workers.__name__
        )
    
    logging.debug(
        f"Participants connected tracked.",
        participants=[w.id for w in workers],
        ID_path=SOURCE_FILE,
        ID_function=connect_to_workers.__name__
    )
    logging.debug(
        f"Local worker's known workers tracked.", 
        known_workers=grid_hook.local_worker._known_workers,
        ID_path=SOURCE_FILE,
        ID_function=connect_to_workers.__name__
    )
    
    return workers


def terminate_connections(ttp, workers):
    """ Terminates the WS connections between remote WebsocketServerWorkers &
        local WebsocketClientWorkers
        Note: Objects do not need to be explicitly cleared since garbage
              collection should kick in by default. Any problems arising from
              this is a sign that something is architecturally wrong with the
              current implementation and should not be silenced.

    Args:
        _id (dict(str)): Processing IDs for caching
    """  
    # Ensure that the grid has it original reference restored. This should
    # destroy all grid references to TTP's VirtualWorker, which is necessary for 
    # it to be successfully deleted
    grid_hook.local_worker = REF_WORKER
    sy.local_worker = REF_WORKER

    # Finally destroy TTP
    ttp.remove_worker_from_local_worker_registry() # remove from ref
    # ttp.remove_worker_from_registry('ttp')         # remove from itself
    del ttp

    try:
        logging.error(
            f"{ttp} has not been deleted!",
            ttp=ttp,
            ID_path=SOURCE_FILE,
            ID_function=connect_to_workers.__name__
        )
    except NameError:
        logging.info(
            "TTP has been successfully deleted!",
            ID_path=SOURCE_FILE,
            ID_function=connect_to_workers.__name__
        )

    for w_idx, worker in enumerate(workers):

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
        del worker

        try:
            logging.error(
                f"Worker_{w_idx} has not been deleted!",
                ID_path=SOURCE_FILE,
                ID_function=connect_to_workers.__name__
            )
        except NameError:
            logging.info(
                f"Worker_{w_idx} has been successfully deleted",
                ID_path=SOURCE_FILE,
                ID_function=connect_to_workers.__name__
            )


def load_selected_run(run_record):
    """ Load in specified federated experimental parameters to be conducted from
        a registered configuration set

    Args:
        run_record (dict): Hyperparameters defining the FL training environment
    Returns:
        FL Training run arguments (Arguments)
    """
    # Remove redundant fields & initialise arguments
    run_params = rpc_formatter.strip_keys(run_record)#, concise=True)
    args = Arguments(**run_params)

    return args


def load_selected_experiment(expt_record):
    """ Load in specified federated model architectures to be used for training
        from configuration files

    Args:
        expt_record (dict): Structural template of model to be initialise
    Returns:
        Model to be used in FL training (Model)
    """
    # Remove redundant fields & initialise Model
    structure = rpc_formatter.strip_keys(expt_record)['model']#, concise=True)
    model = Model(structure)

    return model


def start_expt_run_training(
    keys: dict, 
    action: str,
    registrations: list, 
    experiment: dict, 
    run: dict, 
    auto_align: bool,
    dockerised: bool, 
    log_msgs: bool, 
    verbose: bool
):
    """ Trains a model corresponding to a SINGLE experiment-run combination

    Args:
        keys (dict): Relevant Project ID, Expt ID & Run ID
        action (str): Type of machine learning operation to be executed
        registrations (dict): Registry of all participants for current project
        experiment (dict): Parameters for reconstructing experimental model
        run (dict): Hyperparameters to be used during grid FL training
        auto_align (bool): Toggles if multiple feature alignments will be used
        dockerised (bool): Toggles if current FL grid is containerised or not. 
            If true (default), hosts & ports of all participants are locked at
            "0.0.0.0" & 8020 respectively. Otherwise, participant specified
            configurations will be used (grid architecture has to be finalised).
        log_msgs (bool): Toggles if computation operations should be logged
        verbose (bool): Toggles verbosity of computation logging
    Returns:
        Path-to-trained-models (dict(str))
    """

    def train_combination():

        logging.debug(
            f"Before Initialisation - Reference Worker tracked,",
            ref_worker=REF_WORKER,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Before Initialisation - Local worker w.r.t grid hook tracked.",
            local_worker=grid_hook.local_worker,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Before Initialisation - Local worker w.r.t env tracked.",
            local_worker=sy.local_worker,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        # Create worker representation for local machine as TTP
        ttp = connect_to_ttp(log_msgs=log_msgs, verbose=verbose)

        logging.debug(
            f"After Initialisation - Reference Worker tracked,",
            ref_worker=REF_WORKER,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After Initialisation - Local worker w.r.t grid hook tracked.",
            local_worker=grid_hook.local_worker,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After Initialisation - Local worker w.r.t env tracked.",
            local_worker=sy.local_worker,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        # Complete WS handshake with participants
        workers = connect_to_workers(
            keys=keys,
            reg_records=registrations,
            dockerised=dockerised,
            log_msgs=log_msgs,
            verbose=verbose
        )

        logging.debug(
            f"Before training - Reference Worker tracked,",
            ref_worker=REF_WORKER,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Before training - Registered workers in grid tracked.",
            known_workers=grid_hook.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Before training - Registered workers in env tracked.",
            known_workers=sy.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        ###########################
        # Implementation FootNote #
        ###########################
        
        # Mode customisation will be left here for now. But might be migrated
        # into federated_learning.py if more gpu customisation parameters need
        # to be forwarded into FederatedDataloader. This will localise all GPU
        # integrations into a single file which is easier to maintain

        model = load_selected_experiment(expt_record=experiment)
        if use_gpu and gpu_count > 1:
            model = th.nn.DataParallel(model, device_ids=gpus)
        model = model.to(device)
            
        args = load_selected_run(run_record=run)
    
        # Export trained model weights/biases for persistence
        res_dir = os.path.join(
            out_dir, 
            keys['project_id'], 
            keys['expt_id'], 
            keys['run_id']
        )
        Path(res_dir).mkdir(parents=True, exist_ok=True)

        # Perform a Federated Learning experiment
        fl_expt = FederatedLearning(
            action=action,
            arguments=args, 
            crypto_provider=ttp, 
            workers=workers, 
            reference=model, 
            out_dir=res_dir
        )
        fl_expt.load()
        fl_expt.fit()

        out_paths = fl_expt.export()

        logging.log(
            level=NOTSET,
            event="Final trained model tracked.",
            weights=fl_expt.algorithm.global_model.state_dict(),
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Location of trained model(s) tracked.",
            out_paths=out_paths,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"Loss history tracked.",
            loss_history=fl_expt.algorithm.loss_history,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__    
        )

        logging.debug(
            f"After training - Reference Worker tracked,",
            ref_worker=REF_WORKER,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After training - Registered workers in grid tracked.",
            known_workers=grid_hook.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After training - Registered workers in env tracked.",
            known_workers=sy.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        logging.info(
            f"Model(s) for current federated combination successfully trained and archived!",
            **keys,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        # Close WSCW local objects once training process is completed (if possible)
        # (i.e. graceful termination)
        terminate_connections(ttp=ttp, workers=workers)
        
        logging.debug(
            f"After termination - Reference Worker tracked,",
            ref_worker=REF_WORKER,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After termination - Reference worker's known workers tracked,",
            ref_worker=REF_WORKER._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After termination - Registered workers in grid tracked.",
            known_workers=grid_hook.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )
        logging.debug(
            f"After termination - Registered workers in env tracked.",
            known_workers=sy.local_worker._known_workers,
            ID_path=SOURCE_FILE,
            ID_function=train_combination.__name__
        )

        return out_paths

    logging.info(
        f"Current training combination: {keys}",
        keys=keys,
        ID_path=SOURCE_FILE,
        ID_function=start_expt_run_training.__name__
    )

    # Send initialisation signal to all remote worker WSSW objects
    governor = Governor(auto_align=auto_align, dockerised=dockerised, **keys)
    governor.initialise(reg_records=registrations)

    try:
        results = train_combination()

        def track_object_references(obj):
            """ Tracks the number of references pointing to an object """
            frame = inspect.currentframe()
            try:
                names = [
                    name 
                    for name, val in frame.f_back.f_locals.items() 
                    if val is obj
                ]
                names += [
                    name 
                    for name, val in frame.f_back.f_globals.items()
                    if val is obj and name not in names
                ]
                return names
            finally:
                del frame

        logging.debug(
            "All remaining ObjectPointers un-collected tracked.",
            uncollected_pointers=track_object_references(ObjectPointer),
            ID_path=SOURCE_FILE,
            ID_function=start_expt_run_training.__name__
        )

        logging.info(
            f"Objects left in env: {sy.local_worker._objects}, {sy.local_worker._known_workers}",
            ID_path=SOURCE_FILE,
            ID_function=start_expt_run_training.__name__       
        )
    
    except OSError as o:
        logging.error(
            "Caught an OS problem...",
            description=f"{o}",
            ID_path=SOURCE_FILE,
            ID_function=start_expt_run_training.__name__
        )

    # Send terminate signal to all participants' worker nodes
    # governor = Governor(dockerised=dockerised, **keys)
    governor.terminate(reg_records=registrations)

    return results


# async def train_on_combinations(combinations):
#     """ Asynchroneous function to perform FL grid training over all registered
#         participants over a series of experiment-run combinations

#     Args:
#         combinations (dict(tuple, dict)): All TTP registered combinations
#     Returns:
#         Results of FL grid training from enumerated combinations (dict)
#     """
#     # Apply asynchronous training to each batch
#     futures = [
#         start_expt_run_training(kwargs) 
#         for kwargs in combinations.values()
#     ]
#     all_outpaths = await asyncio.gather(*futures)

#     results = dict(zip(combinations.keys(), all_outpaths))
#     return results


def start_proc(kwargs):
    """ Automates the execution of Federated learning experiments on different
        hyperparameter sets & model architectures

    Args:
        kwargs (dict): Experiments & models to be tested
    Returns:
        Path-to-trained-models (list(str))
    """
    action = kwargs['action']
    experiments = kwargs['experiments']
    runs = kwargs['runs']
    registrations = kwargs['registrations']
    is_verbose = kwargs['verbose']
    log_msgs = kwargs['log_msgs']
    is_dockerised = kwargs['dockerised']
    auto_align = kwargs['auto_align']

    training_combinations = {}
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
                    'action': action,
                    'registrations': registrations,
                    'experiment': expt_record,
                    'run': run_record,
                    'auto_align': auto_align,
                    'dockerised': is_dockerised, 
                    'log_msgs': log_msgs, 
                    'verbose': is_verbose
                }
                training_combinations[combination_key] = project_expt_run_params

    logging.info(
        "Training combinations loaded!",
        training_combinations=f"{training_combinations}",
        ID_path=SOURCE_FILE,
        ID_function=start_proc.__name__
    )

    # """
    # ##########################################################
    # # Multiprocessing - Run each process on an isolated core #
    # ##########################################################
    # [Redacted - Due to PySyft objects being unserialisable, this feature has
    #             been frozen until further notice
    # ]

    # pool = ProcessingPool(nodes=core_count)

    # results = pool.amap(start_expt_run_training, training_combinations.values())

    # while not results.ready():
    #     time.sleep(1)
        
    # results = results.get()

    # # Clean up Pathos multiprocessing pool
    # pool.close()
    # pool.join()
    # pool.terminate()

    # completed_trainings = dict(zip(training_combinations.keys(), results))
    # return completed_trainings
    # """

    # """
    # #############################################################
    # # Optimisation Alternative - Asynchronised FL grid Training #
    # #############################################################
    # [Redacted - Due to _recv stuck listening in a circular loop, this feature 
    #             has been frozen until further notice
    # ]

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    # try:
    #     completed_trainings = loop.run_until_complete(
    #         train_on_combinations(combinations=training_combinations)
    #     )
    # finally:
    #     loop.close()

    # return completed_trainings
    # """
    
    # #######################################################################
    # # Default Implementation - Synchroneous execution of FL grid training #
    # #######################################################################

    # # results = map(
    # #     lambda kwargs: start_expt_run_training(**kwargs), 
    # #     training_combinations.values()
    # # )

    completed_trainings = {
        combination_key: start_expt_run_training(**kwargs) 
        for combination_key, kwargs in training_combinations.items()
    }

    return completed_trainings
