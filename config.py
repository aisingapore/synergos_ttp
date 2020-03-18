#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import json
import logging
import os
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path

# Libs
import numpy as np
import torch as th
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_smartcache import SmartCacheTable

# Custom
from core import DateTimeSerializer, TimeDeltaSerializer

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

SRC_DIR = Path(__file__).parent.absolute()

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
    config_globstring = os.path.join(SRC_DIR, dirname, "*.json")
    config_paths = glob(config_globstring)

    return {parse_filename(c_path): c_path for c_path in config_paths}

###########################
# PySyft Template Schemas #
###########################
"""
For REST service to be stable, there must be schemas enforced to ensure that any
erroneous queries will affect the functions of the system.
"""
template_paths = detect_configurations("templates")

schemas = {}
for name, s_path in template_paths.items():
    with open(s_path, 'r') as schema:
        schemas[name] = json.load(schema, object_pairs_hook=OrderedDict)

logging.debug(f"Schemas loaded: {list(schemas.keys())}")

##########################################
# PySyft Project Database Configurations #
##########################################
""" 
In PySyft TTP, each registered project is granted its own database, where 2 
tables reside - Participant & Experiment. All interaction between Project,
Participant, Experiment & Runs must conform to specified template schemas. Refer
to the `templates` directory for the actual schemas.
"""
db_path = os.path.join(SRC_DIR, "data", "database.json")

serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
serialization.register_serializer(TimeDeltaSerializer(), 'TinyDelta')

database = TinyDB(
    path=db_path, 
    sort_keys=True,
    indent=4,
    separators=(',', ': '),
    default_table="Project",
    storage=CachingMiddleware(serialization)
)

#database.table_class = SmartCacheTable

logging.debug(f"Project Database loaded: {database}")

#########################################
# PySyft Container Local Configurations #
#########################################

server_params = {
    
    # Define server's role: Master or slave
    'is_master': True,
    
    # State input directory
    'in_dir': os.path.join(SRC_DIR, "inputs"),

    # State output directory
    'out_dir': os.path.join(SRC_DIR, "outputs"),

    # Initialise Cache
    'cache': infinite_nested_dict()

}

logging.debug(f"Server configurations: {server_params}")