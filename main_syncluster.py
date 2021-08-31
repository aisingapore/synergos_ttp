#!/usr/bin/env python

####################
# Required Modules #
####################

# from gevent import monkey
# monkey.patch_all()

# Generic/Built-in
import argparse
import logging
import os
import time
import uuid
from typing import Dict, List, Tuple, Callable, Union, Any

# Libs
import ray

# Custom
import config
from config import (
    SRC_DIR, RETRY_INTERVAL,
    capture_system_snapshot,
    configure_grid,
    configure_synergos_variant,
    configure_node_logger, 
    configure_sysmetric_logger,
    count_available_cpus,
    count_available_gpus
)
from synmanager.preprocess_operations import PreprocessConsumerOperator
from synmanager.train_operations import TrainConsumerOperator
from synmanager.evaluate_operations import EvaluateConsumerOperator
from synmanager.completed_operations import CompletedProducerOperator

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

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
    optimize_job: Callable,
    validate_job: Callable,
    predict_job: Callable,
    **kwargs
):
    """ Run endless loop to poll messages across different queues and run 
        respective callback operations. The hierachy of priority between queues, 
        starting from the highest is as follows: 
            Preprocess -> Train -> Evaluate (Validation or Prediction)

    Args:
        host (str): IP of server where queue is hosted on
        port (int): Port of server allocated to queue
        preprocess_job (Callable): Function to execute when handling a polled
            preprocessing job from the "Preprocess" queue
        train_job (Callable): Function to execute when handling a polled
            training job from the "Train" queue
        validate_job (Callable): Function to execute when handling a polled
            validation job from the "Evaluate" queue
        predict_job (Callable): Function to execute when handling a polled
            prediction job from the "Evaluate" queue
    """
    def executable_job(
        process: str, 
        keys: List[str],
        grids: List[Dict[str, Any]],
        parameters: Dict[str, Union[str, int, float, list, dict]]
    ) -> Callable:
        """
        
        Args:
            process (str):
            combination_key (tuple):
            combination_params (dict):
        """
        JOB_MAPPINGS = {
            'preprocess': preprocess_job,
            'train': train_job,
            'optimize': optimize_job,
            'validate': validate_job,
            'predict': predict_job
        }

        selected_job = JOB_MAPPINGS[process]
        info = selected_job(keys, grids, parameters)
        return {'process': process, **info}

    logger = kwargs.get('logger', logging)

    preprocess_consumer = PreprocessConsumerOperator(host=host, port=port)
    train_consumer = TrainConsumerOperator(host=host, port=port)
    evaluate_consumer = EvaluateConsumerOperator(host=host, port=port)

    completed_producer = CompletedProducerOperator(host=host, port=port)

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

        preprocess_consumer.connect()
        train_consumer.connect()
        evaluate_consumer.connect()
        completed_producer.connect()

        while True:

            # try:
            # Check message count in higher priority queues
            preprocess_messages = preprocess_consumer.check_message_count()
            train_messages = train_consumer.check_message_count()
            evaluate_messages = evaluate_consumer.check_message_count()

            if preprocess_messages > 0:
                job_info = preprocess_consumer.poll_message(executable_job)

            elif train_messages > 0:
                job_info = train_consumer.poll_message(executable_job)

            elif evaluate_messages > 0:
                job_info = evaluate_consumer.poll_message(executable_job)

            else:
                logger.synlog.info(
                    f"No jobs in queue! Waiting for {RETRY_INTERVAL} second...",
                    ID_path=os.path.join(SRC_DIR, "config.py"), 
                    ID_function=poll_cycle.__name__
                )
                job_info = None

            if job_info:
                logging.warning(f"--->>> Completed info: {job_info}") 
                completed_producer.process(**job_info)
            
            # except Exception as e:
            #     logger.synlog.error(
            #         f"Something went wrong while running a job! Error: {e}",
            #         ID_path=os.path.join(SRC_DIR, "config.py"), 
            #         ID_function=poll_cycle.__name__
            #     )

            time.sleep(RETRY_INTERVAL)
    
    except KeyboardInterrupt:
        logger.synlog.info(
            "[Ctrl-C] recieved! Job polling terminated.",
            ID_path=os.path.join(SRC_DIR, "config.py"), 
            ID_function=poll_cycle.__name__
        )

    finally:
        preprocess_consumer.disconnect()
        train_consumer.disconnect()
        evaluate_consumer.disconnect()
        completed_producer.disconnect()

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
        "--queue",
        "-mq",
        type=str,
        default=["rabbitmq"],
        nargs="+",
        help="Type of queue framework to use. eg. --queue rabbitmq 127.0.0.1 5672"
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

    # Activate Synergos SynCluster variant
    configure_synergos_variant(is_cluster=True)

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

    # Bind node to queue
    mq_kwargs = construct_queue_kwargs(**input_kwargs)
    node_logger.synlog.info(
        f"Orchestrator `{server_id}` -> Snapshot of Queue Parameters",
        **mq_kwargs
    )

    # Set up sysmetric logger
    sysmetric_logger = configure_sysmetric_logger(**logger_kwargs)
    sysmetric_logger.track(
        file_path=SOURCE_FILE,
        class_name="",
        function_name=""        
    )

    ###########################
    # Implementation Footnote #
    ###########################

    # [Cause]
    # In SynCluster mode, all processes are inducted as jobs. All jobs are sent
    # to Synergos MQ to be linearized for parallel distributed computing.

    # [Problems]
    # 

    # [Solution]
    # Start director as a ray head node, with all other TTPs as child nodes 
    # connecting to it. Tuning parameters will be reported directly to the head
    # node, bypassing the queue

    ray.init()
    assert ray.is_initialized() == True

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

    # Apply custom configurations on server
    from rest_rpc import initialize_app   
    initialize_app(settings=config)

    from rest_rpc.training.alignments import execute_alignment_job
    from rest_rpc.training.models import execute_training_job
    from rest_rpc.training.optimizations import execute_optimization_job
    from rest_rpc.evaluation.validations import execute_validation_job
    from rest_rpc.evaluation.predictions import execute_prediction_job
        
    try:
        # Commence job polling cycle 
        poll_cycle(
            **mq_kwargs,
            preprocess_job=execute_alignment_job,
            train_job=execute_training_job,
            optimize_job=execute_optimization_job,
            validate_job=execute_validation_job,
            predict_job=execute_prediction_job,
            logger=node_logger
        )

    finally:
        sysmetric_logger.terminate()

        ray.shutdown()
        assert ray.is_initialized() == False