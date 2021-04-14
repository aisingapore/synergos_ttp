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
import uuid
from pathlib import Path

# Libs
import ray

# Custom
import config
from config import (
    capture_system_snapshot,
    configure_node_logger, 
    configure_sysmetric_logger,
    count_available_cpus,
    count_available_gpus
)

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

SECRET_KEY = "synergos_ttp" #os.urandom(24) # secret key

#############
# Functions #
#############

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

    ### No need to configure Synergos variant since Basic is default ###

    ### No need to configure grid since only 1 grid -> Grid Idx is 0 ###

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
    sysmetric_logger.track(
        file_path=SOURCE_FILE,
        class_name="",
        function_name=""
    )

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

    from rest_rpc import initialize_app
        
    ray.init(
        local_mode=False, 
        dashboard_host="0.0.0.0", 
        dashboard_port=8080,
        num_cpus=count_available_cpus(),
        # num_gpus=count_available_gpus()
    )

    try:

        # cache = app.config['CACHE']
        # cache.init_app(
        #     app=app, 
        #     config={
        #         "CACHE_TYPE": "filesystem",
        #         'CACHE_DIR': Path('/worker/tmp')
        #     }
        # )

        app = initialize_app(settings=config)
        app.run(host="0.0.0.0", port=5000)

    finally:
        sysmetric_logger.terminate()

        ray.shutdown()