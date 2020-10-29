#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import json
import logging
import os
import random
import subprocess
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path

# Libs
import numpy as np
import torch as th
import psutil

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

SRC_DIR = Path(__file__).parent.absolute()

API_VERSION = "0.1.0"

####################
# Helper Functions #
####################

def seed_everything(seed=42):
    """ Convenience function to set a constant random seed for model consistency

    Args:
        seed (int): Seed for RNG
    Returns:
        True    if operation is successful
        False   otherwise
    """
    try:
        random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return True

    except:
        return False


def count_available_gpus() -> int:
    """ Counts no. of attached GPUs devices in the current system. As GPU 
        support is supplimentary, if any exceptions are caught here, system
        defaults back to CPU-driven processes (i.e. gpu count is 0)

    Returns:
        gpu_count (int)
    """
    try:
        process = subprocess.run(
            ['lspci'],
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        all_detected_devices = process.stdout.split('\n')
        gpus = [
            device 
            for device in all_detected_devices 
            if (('VGA' in device) or ('Display' in device)) and
            'Integrated Graphics' not in device # exclude integrated graphics
        ]
        logging.debug(f"Detected GPUs: {gpus}")
        return len(gpus)

    except subprocess.CalledProcessError as cpe:
        logging.warning(f"Could not detect GPUs! Error: {cpe}")
        logging.warning(f"Defaulting to CPU processing instead...")
        return 0
        

def detect_configurations(dirname):
    """ Automates loading of configuration files in specified directory

    Args:
        dirname (str): Target directory to load configurations from
    Returns:
        Params (dict)
    """

    def parse_filename(filepath):
        """ Extracts filename from a specified filepath
            Assumptions: There are no '.' in filename
        
        Args:
            filepath (str): Path of file to parse
        Returns:
            filename (str)
        """
        return os.path.basename(filepath).split('.')[0]

    # Load in parameters for participating servers
    config_globstring = os.path.join(SRC_DIR, dirname, "**/*.json")
    config_paths = glob(config_globstring)

    return {parse_filename(c_path): c_path for c_path in config_paths}

#############################################
# PySyft TTP Container Local Configurations #
#############################################
""" 
General parameters required for processing inputs & outputs
"""

# Define server's role: Master or slave
IS_MASTER = True

# State input directory
IN_DIR = os.path.join(SRC_DIR, "inputs")

# State output directory
OUT_DIR = os.path.join(SRC_DIR, "outputs")

# State data directory
DATA_DIR = os.path.join(SRC_DIR, "data")

# State test directory
TEST_DIR = os.path.join(SRC_DIR, "tests")

# State MLFlow local directory
MLFLOW_DIR = os.path.join(SRC_DIR, "mlflow")

# Initialise Cache
CACHE = infinite_nested_dict()

# Allocate no. of cores for processes
CORES_USED = psutil.cpu_count(logical=True) - 1

# Detect no. of GPUs attached to server
GPU_COUNT = count_available_gpus()

logging.debug(f"Is master node? {IS_MASTER}")
logging.debug(f"Input directory detected: {IN_DIR}")
logging.debug(f"Output directory detected: {OUT_DIR}")
logging.debug(f"Data directory detected: {DATA_DIR}")
logging.debug(f"Test directory detected: {TEST_DIR}")
logging.debug(f"MLFlow directory detected: {MLFLOW_DIR}")
logging.debug(f"Cache initialised: {CACHE}")
logging.debug(f"No. of available CPU Cores: {CORES_USED}")
logging.debug(f"No. of available GPUs: {GPU_COUNT}")

##########################################
# PySyft Project Database Configurations #
##########################################
""" 
In PySyft TTP, each registered project is factored into many tables, namely 
Project, Experiment, Run, Participant, Registration, Tag, Alignment & Model, all
related hierarchially. All interactions must conform to specified relation & 
association rules. Refer to the Record classes in all `rest_rpc/*/core/utils.py`
for more detailed descriptions of said relations/associations.

Also, all archived payloads must conform to specified template schemas. Refer 
to the `templates` directory for the actual schemas.
"""
DB_PATH = os.path.join(SRC_DIR, "data", "database.json")

logging.debug(f"Database path detected: {DB_PATH}")

###############################
# PySyft TTP Template Schemas #
###############################
"""
For REST service to be stable, there must be schemas enforced to ensure that any
erroneous queries will affect the functions of the system.
"""
template_paths = detect_configurations("templates")

SCHEMAS = {}
for name, s_path in template_paths.items():
    with open(s_path, 'r') as schema:
        SCHEMAS[name] = json.load(schema, object_pairs_hook=OrderedDict)

logging.debug(f"Schemas loaded: {list(SCHEMAS.keys())}")

#######################################
# PySyft Flask Payload Configurations #
####################################### 
"""
Responses for REST-RPC have a specific format to allow compatibility between TTP
& Worker Flask Interfaces. Remember to modify rest_rpc.connection.core.utils.Payload 
upon modifying this template!
"""
PAYLOAD_TEMPLATE = {
    'apiVersion': API_VERSION,
    'success': 0,
    'status': None,
    'method': "",
    'params': {},
    'data': {}
}

#################################
# PySyft REST-RPC Worker Routes #
#################################
"""
In a PySyft REST-RPC Worker Node, there are a few flask routes that serve as
interfacing services in order to initialise the WSSW pysyft worker.
"""
WORKER_ROUTES = {
    'poll': '/worker/poll/<project_id>',
    'align': '/worker/align/<project_id>',
    'initialise': '/worker/initialise/<project_id>/<expt_id>/<run_id>',
    'terminate': '/worker/terminate/<project_id>/<expt_id>/<run_id>',
    'predict': '/worker/predict/<project_id>/<expt_id>/<run_id>'
}