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
from string import Template

# Libs
import numpy as np
import psutil
import torch as th

# Custom
from synlogger.general import TTPLogger, SysmetricLogger

##################
# Configurations #
##################

SRC_DIR = Path(__file__).parent.absolute()

API_VERSION = "0.2.0"

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

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


def count_available_cpus(safe_mode: bool = False, r_count: int = 1) -> int:
    """ Counts no. of detected CPUs in the current system. By default, all 
        CPU cores detected are returned. However, if safe mode is toggled, then
        a specified number of cores are reserved.
    
    Args:
        safe_mode (bool): Toggles if cores are reserved
        r_count (int): No. of cores to reserve
    Return:
        No. of usable cores (int)
    """
    total_cores_available = psutil.cpu_count(logical=True)
    reserved_cores = safe_mode * r_count
    return total_cores_available - reserved_cores


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
            if (('VGA' in device) or ('Display' in device)) 
            and 'Integrated Graphics' not in device # exclude integrated graphics
            and 'Intel' not in device               # Catch edge cases (hack)
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


def capture_system_snapshot() -> dict:
    """ Takes a snapshot of parameters used in system-wide operations

    Returns:
        System snapshot (dict)
    """
    return {
        'IS_CLUSTER': IS_CLUSTER,
        'IS_MASTER': IS_MASTER,
        'GRID': GRID,
        'IN_DIR': IN_DIR,
        'OUT_DIR': OUT_DIR,
        'DATA_DIR': DATA_DIR,
        'MLFLOW_DIR': MLFLOW_DIR,
        'TEST_DIR': TEST_DIR,
        'CORES_USED': CORES_USED,
        'GPU_COUNT': GPU_COUNT,
        'GPUS': GPUS,
        'USE_GPU': USE_GPU,
        'DEVICE': DEVICE,
        'DB_PATH': DB_PATH,
        'SCHEMAS': SCHEMAS,
        'RETRY_INTERVAL': RETRY_INTERVAL
    }


def configure_grid(grid: int) -> int:
    """ Binds the server to a specific grid referenced by its index. This is
        important when running the SynCluster configuration of Synergos.

    Args:
        grid (int): Grid to be bounded to
    Returns:
        Bounded grid (int)
    """
    global GRID 
    GRID = grid
    return GRID


def configure_synergos_variant(is_cluster: bool) -> int:
    """ Defines which configuration of Synergos to run (i.e. Basic/SynCluster).
        This is important because this toggles the usage of queues for 
        parallellized workflows

    Args:
        is_cluster (bool): Toggles if TTP is in Basic or SynCluster mode
    Returns:
        TTP state (bool)
    """
    global IS_CLUSTER, IS_MASTER
    IS_CLUSTER = is_cluster

    if is_cluster:
        IS_MASTER = False

    return IS_CLUSTER 


def configure_cpu_allocation(**res_kwargs) -> int:
    """ Configures no. of CPU cores available to the system. By default, all
        CPU cores will be allocated.

    Args:
        res_kwargs: Any custom resource allocations declared by user
    Returns:
        CPU cores used (int) 
    """
    global CORES_USED
    cpu_count = res_kwargs.get('cpus')
    CORES_USED = min(cpu_count, CORES_USED) if cpu_count else CORES_USED
    return CORES_USED


def configure_gpu_allocation(**res_kwargs):
    """ Configures no. of GPU cores available to the system.

    Args:
        res_kwargs: Any custom resource allocations declared by user
    Returns:
        GPU cores used (int) 
    """
    global GPU_COUNT
    gpu_count = res_kwargs.get('gpus')
    GPU_COUNT = min(gpu_count, GPU_COUNT) if gpu_count else GPU_COUNT
    return GPU_COUNT


def configure_node_logger(**logger_kwargs) -> TTPLogger:
    """ Initialises the synergos logger corresponding to the current node type.
        In this case, a TTPLogger is initialised.

    Args:
        logger_kwargs: Any parameters required for node logger configuration
    Returns:
        Node logger (TTPLogger)
    """
    global NODE_LOGGER
    NODE_LOGGER = TTPLogger(**logger_kwargs)
    NODE_LOGGER.initialise()
    return NODE_LOGGER


def configure_sysmetric_logger(**logger_kwargs) -> SysmetricLogger:
    """ Initialises the sysmetric logger to facillitate polling for hardware
        statistics.

    Args:
        logger_kwargs: Any parameters required for node logger configuration
    Returns:
        Sysmetric logger (SysmetricLogger)
    """
    global SYSMETRIC_LOGGER
    SYSMETRIC_LOGGER = SysmetricLogger(**logger_kwargs)
    return SYSMETRIC_LOGGER

########################################################
# Synergos Orchestrator Container Local Configurations #
########################################################
""" 
General parameters required for processing inputs & outputs
"""

# Define deployment configuration
IS_CLUSTER = False  # default: Synergos Basic

# Define server's role: Master or slave
IS_MASTER = True    # default: Synergos Basic -> non-cluster mode -> Orchestrator

# State grid server is bounded to
GRID = 0            # default: Synergos Basic -> only 1 grid -> grid idx is 0

# State input directory
IN_DIR = os.path.join(SRC_DIR, "inputs")

# State output directory
OUT_DIR = os.path.join(SRC_DIR, "outputs")

# State data directory
DATA_DIR = os.path.join(SRC_DIR, "data")

# State test directory
TEST_DIR = os.path.join(SRC_DIR, "tests")

# State MLFlow local directory
MLFLOW_DIR = "/mlflow"

# Initialise Cache
CACHE = infinite_nested_dict()

# Allocate no. of cores for processes
CORES_USED = count_available_cpus(safe_mode=True)

# Detect no. of GPUs attached to server
GPU_COUNT = count_available_gpus()
GPUS = [g_idx for g_idx in range(GPU_COUNT)]
USE_GPU = GPU_COUNT > 0 and th.cuda.is_available()
DEVICE = th.device('cuda' if USE_GPU else 'cpu')

# Retry interval for contacting idle workers
RETRY_INTERVAL = 5  # in seconds

logging.debug(f"Grid linked: {GRID}")
logging.debug(f"Is master node? {IS_MASTER}")
logging.debug(f"Input directory detected: {IN_DIR}")
logging.debug(f"Output directory detected: {OUT_DIR}")
logging.debug(f"Data directory detected: {DATA_DIR}")
logging.debug(f"Test directory detected: {TEST_DIR}")
logging.debug(f"MLFlow directory detected: {MLFLOW_DIR}")
logging.debug(f"Cache initialised: {CACHE}")
logging.debug(f"No. of available CPU Cores: {CORES_USED}")
logging.debug(f"No. of available GPUs: {GPU_COUNT}")
logging.debug(f"Are GPUs active: {USE_GPU}")
logging.debug(f"Final device used: {DEVICE}")
logging.debug(f"Retry Interval: {RETRY_INTERVAL} seconds")

#############################################
# Synergos Metadata Database Configurations #
#############################################
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

#########################################
# Synergos Marshalling Template Schemas #
#########################################
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

########################################
# Synergos REST Payload Configurations #
######################################## 
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

##########################################
# Synergos Worker Logging Configurations #
##########################################
"""
Synergos has certain optional components, such as a centrialised logging 
server, as well as a metadata catalogue. This section governs configuration of 
the orchestrator node to facilitate such integrations, where applicable. This 
portion gets configured during runtime. By default, unconfigured node &
sysmetric loggers are loaded.
"""
NODE_LOGGER = configure_node_logger(logger_name=f"ttp_{GRID}")
SYSMETRIC_LOGGER = configure_sysmetric_logger(logger_name=f"ttp_{GRID}")

###################################
# Synergos REST-RPC Worker Routes #
###################################
"""
In a Synergos REST-RPC Worker Node, there are a few flask routes that serve as
interfacing services in order to initialise the WSSW pysyft worker.
"""
WORKER_ROUTE_TEMPLATES = {
    'poll': Template('/worker/poll/$collab_id/$project_id'),
    'align': Template('/worker/align/$collab_id/$project_id'),
    'initialise': Template('/worker/initialise/$collab_id/$project_id/$expt_id/$run_id'),
    'terminate': Template('/worker/terminate/$collab_id/$project_id/$expt_id/$run_id'),
    'predict': Template('/worker/predict/$collab_id/$project_id/$expt_id/$run_id')
}

NODE_ID_TEMPLATE = Template("$participant") #Template("$participant-[$node]")
NODE_PID_REGEX = "^(.*)(?=-\[node_\d*\])"
NODE_NID_REGEX = "(?:(?!\[)(node_\d*)(?=\]$))"