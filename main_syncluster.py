#!/usr/bin/env python

####################
# Required Modules #
####################

# from gevent import monkey
# monkey.patch_all()

# Generic/Built-in
import argparse
import logging
from random import randint
from rest_rpc.evaluation.core.server import evaluate_proc
import time
import uuid
from typing import Callable

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
from synmanager.preprocess_operations import PreprocessConsumerOperator
from synmanager.train_operations import TrainConsumerOperator
from synmanager.evaluate_operations import EvaluateConsumerOperator

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
            f"Specified logging variant '{logging_variant}' is not supported!"
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


def construct_queue_kwargs(**kwargs):
    """ Extracts queue configuration values for linking server to said queue

    Args:
        kwargs: Any user input captured 
    Returns:
        Queue configurations (dict)
    """
    queue_config = kwargs['queue']

    queue_variant = queue_config[0]
    if queue_variant not in ["rabbitmq"]:
        raise argparse.ArgumentTypeError(
            f"Specified queue variant '{queue_variant}' is not supported!"
        )

    server = queue_config[1]
    port = int(queue_config[2])

    return {'host': server, 'port': port}


def poll_cycle(
    host: str,
    port: int,
    preprocess_job: Callable,
    train_job: Callable,
    validate_job: Callable,
    predict_job: Callable
):
    """ Run endless loop to poll messages across different queues and run 
        respective callback operations. The hierachy of priority between queues, 
        starting from the highest is as follows: 
            Preprocess -> Train -> Evaluate
    """
    def executable_job(process, combination_key, combination_params):
        """
        """
        JOB_MAPPINGS = {
            'preprocess': preprocess_job,
            'train': train_job,
            'validate': validate_job,
            'predict': predict_job
        }

        selected_job = JOB_MAPPINGS[process]
        return selected_job(combination_key, combination_params)

    preprocess_consumer = PreprocessConsumerOperator(host=host, port=port)
    train_consumer = TrainConsumerOperator(host=host, port=port)
    evaluate_consumer = EvaluateConsumerOperator(host=host, port=port)

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

    while True:

        # Check message count in higher priority queues
        preprocess_messages = preprocess_consumer.check_message_count()
        train_messages = train_consumer.check_message_count()
        evaluate_messages = evaluate_consumer.check_message_count()

        try:
            if preprocess_messages > 0:
                preprocess_consumer.poll_message(executable_job)

            elif train_messages > 0:
                train_consumer.poll_message(executable_job)

            elif evaluate_messages > 0:
                evaluate_consumer.poll_message(executable_job)

            else:
                print("No jobs in queue! Waiting for 1 second...")
        
        except Exception:
            pass

        time.sleep(1)

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
        default=["basic"],
        nargs="+",
        help="Type of logging framework to use. eg. --logging_variant graylog 127.0.0.1 9400"
    )

    parser.add_argument(
        "--queue",
        "-mq",
        type=str,
        default=["rabbitmq"],
        nargs="+",
        help="Type of queue framework to use. eg. --queue rabbitmq 127.0.0.1 5672"
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

    mq_kwargs = construct_queue_kwargs(**input_kwargs)
    node_logger.synlog.info(
        f"Orchestrator `{server_id}` -> Snapshot of Queue Parameters",
        **mq_kwargs
    )

    # # Set up sysmetric logger
    # sysmetric_logger = configure_sysmetric_logger(**logger_kwargs)
    # sysmetric_logger.track("/test/path", 'TestClass', 'test_function')


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

    from rest_rpc.training.alignments import execute_alignment_job
    from rest_rpc.training.models import execute_training_job
    # from rest_rpc.evaluation.validations import execute_validation
    # from rest_rpc.evaluation.predictions import execute_prediction
        
    try:
        poll_cycle(
            **mq_kwargs,
            preprocess_job=execute_alignment_job,
            train_job=execute_training_job,
            evaluate_job=evaluate_inference_job
        )

    finally:
        # sysmetric_logger.terminate()
        pass
