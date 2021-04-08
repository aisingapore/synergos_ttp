#!/usr/bin/env python

####################
# Required Modules #
####################

# from gevent import monkey
# monkey.patch_all()

# Generic/Built-in
import argparse
import logging
import uuid
from pathlib import Path

# Libs
import ray

# Custom
from config import (
    capture_system_snapshot,
    configure_grid,
    configure_node_logger, 
    configure_sysmetric_logger,
    count_available_cpus,
    count_available_gpus
)
from synmanager.train import 

##################
# Configurations #
##################

SECRET_KEY = "synergos_ttp" #os.urandom(24) # secret key

#############
# Functions #
#############

def construct_grid_kwargs(**kwargs) -> dict:
    """ Extracts grid configuration values for linking server to said grid

    Args:
        kwargs: Any user input captured 
    Returns:
        Grid configurations (dict)
    """
    return {'grid': kwargs['grid']}


def construct_logger_kwargs(**kwargs) -> dict:
    """ Extracts user-parsed values and re-mapping them into parameters 
        corresponding to those required of components from Synergos Logger.

    Args:
        kwargs: Any user input captured 
    Returns:
        Logger configurations (dict)
    """
    logger_name = kwargs['id']

    logging_config = kwargs['logging_variant']

    logging_variant = logging_config[0]
    if logging_variant not in ["basic", "graylog"]:
        raise argparse.ArgumentTypeError(
            f"Specified variant '{logging_variant}' is not supported!"
        )

    server = (logging_config[1] if len(logging_config) > 1 else None)
    port = (int(logging_config[2]) if len(logging_config) > 1 else None)

    debug_mode = kwargs['debug']
    logging_level = logging.DEBUG if debug_mode else logging.INFO
    debugging_fields = debug_mode

    is_censored = kwargs['censored']
    censor_keys = (
        [
            'SRC_DIR', 'IN_DIR', 'OUT_DIR', 'DATA_DIR', 'MODEL_DIR', 
            'CUSTOM_DIR', 'TEST_DIR', 'DB_PATH', 'CACHE_TEMPLATE', 
            'PREDICT_TEMPLATE'
        ]
        if is_censored 
        else []
    )

    return {
        'logger_name': logger_name,
        'logging_variant': logging_variant,
        'server': server,
        'port': port,
        'logging_level': logging_level,
        'debugging_fields': debugging_fields,
        'censor_keys': censor_keys
    }

# def str2bool(v):
#     if isinstance(v, bool):
#        return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# def poll_cycle(host):
#     '''
#     Run endless loop to poll messages across different queues and run respective callback operations.
#     The hierachy of priority between queues, starting from the highest: Preprocess -> Train -> Evaluate
#     '''
#     i = 1
#     preprocess_consume = PreprocessConsumerOperator(host)
#     train_consume = TrainConsumerOperator(host)
#     evaluate_consume = EvaluateConsumerOperator(host)
    
#     while True:

#         if (i%3 == 1):
#             # Primary priority - preprocess queue
#             preprocess_consume.poll_message(start_alignment)

#         elif (i%3 == 2):
#             # Secondary priority - train queue

#             # Check message count in higher priority queue
#             preprocess_messages = preprocess_consume.check_message_count()
#             if ( preprocess_messages > 0 ):
#                 i = 1
#                 sleep(3)
#                 continue
#             else:
#                 train_consume.poll_message(start_training)

#         else:
#             # Tertiary priority - tertiary queue

#             # Check message count in higher priority queues
#             preprocess_messages = preprocess_consume.check_message_count()
#             train_messages = train_consume.check_message_count()
            
#             if (preprocess_messages > 0) or (train_messages > 0):
#                 i = 1
#                 sleep(3)
#                 continue
#             else:
#                 evaluate_consume.poll_message(start_evaluation)

#         sleep(3)
#         i += 1

###########
# Scripts #
###########

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="REST-RPC Orchestrator for a Synergos Network."
    )

    parser.add_argument(
        "--id",
        "-i",
        type=str,
        default=f"ttp/{uuid.uuid4()}",
        help="ID of orchestrating party. e.g. --id TTP"
    )

    parser.add_argument(
        "--grid",
        "-g",
        type=int,
        default=0,
        help="Grid index that this Synergos TTP node is bounded on."
    )

    parser.add_argument(
        "--logging_variant",
        "-l",
        type=str,
        default="basic",
        nargs="+",
        help="Type of logging framework to use. eg. --logging_variant graylog 127.0.0.1 9400"
    )

    parser.add_argument(
        '--censored',
        "-c",
        action='store_true',
        default=False,
        help="Toggles censorship of potentially sensitive information on this orchestrator node"
    )

    parser.add_argument(
        '--debug',
        "-d",
        action='store_true',
        default=False,
        help="Toggles debugging mode on this orchestrator node"
    )

    input_kwargs = vars(parser.parse_args())

    # Bind node to grid
    grid_kwargs = construct_grid_kwargs(**input_kwargs)
    configure_grid(**grid_kwargs)

    # Set up core logger
    server_id = input_kwargs['id']
    logger_kwargs = construct_logger_kwargs(**input_kwargs)
    node_logger = configure_node_logger(**logger_kwargs)
    node_logger.synlog.info(
        f"Orchestrator `{server_id}` -> Snapshot of Input Parameters",
        **input_kwargs
    )
    node_logger.synlog.info(
        f"Orchestrator `{server_id}` -> Snapshot of Logging Parameters",
        **logger_kwargs
    )

    system_kwargs = capture_system_snapshot()
    node_logger.synlog.info(
        f"Orchestrator `{server_id}` -> Snapshot of System Parameters",
        **system_kwargs
    )

    # Set up sysmetric logger
    sysmetric_logger = configure_sysmetric_logger(**logger_kwargs)
    sysmetric_logger.track("/test/path", 'TestClass', 'test_function')


    ###########################
    # Implementation Footnote #
    ###########################

    # [Cause]
    # To allow custom Synergos Logging components to permeate the entire
    # system, these loggers have to be initialised first before the system
    # performs module loading. 

    # [Problems]
    # Importing app right at the start of the page causes system modules to
    # be loaded first, resulting in AttributeErrors, since 
    # synlogger.general.WorkerLogger has not been intialised, and thus, its
    # corresponding `synlog` attribute cannot be referenced.

    # [Solution]
    # Import system modules only after loggers have been intialised.

    from rest_rpc.training.core.server import align_proc, train_proc
    from rest_rpc.evaluation.core.server import evaluate_proc
        
    try:
        ###########################
        # Implementation Footnote #
        ###########################

        # [Cause]
        # When running Synergos in SynCluster mode, TTPs, while still 
        # commanding single grids, are no longer master orchestration nodes,
        # and will receive orchestration instructions from the Director 

        # [Problems]
        # There should be no active REST interface on any TTP.

        # [Solution]
        # Replace REST interface with queue listeners for the Preprocess, Train
        # & Evaluate queues in Synergos MQ to allow TTPs to retrieve jobs to 
        # run in their respective local grids
        
        pass

    finally:
        sysmetric_logger.terminate()
