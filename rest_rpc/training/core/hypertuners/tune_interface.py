#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import uuid
import argparse
from typing import Dict, List, Tuple, Union

# Libs
import ray
from ray import tune

# Custom
from rest_rpc import app
from rest_rpc.connection.core.utils import (
    RunRecords
)
from rest_rpc.training.core.hypertuners import AbstractTuner
from rest_rpc.training.core.hypertuners.tune_driver_script import start_generate_hp, start_hp_training

##################
# Configurations #
##################

db_path = app.config['DB_PATH']
run_records = RunRecords(db_path=db_path)

########################################
# HP Tuning Class - HPTuning #
########################################
class RayTuneTuner(AbstractTuner):

    def __init__(self):
        pass

    def tune(
        self,
        project_id: str,
        expt_id: str,
        search_space: Dict[str, Dict[str, Union[str, bool, int, float]]],
        n_samples: int = 3
    ):
        kwargs = {
            "project_id": project_id,
            "expt_id": expt_id,
            "n_samples": n_samples,
            "search_space": search_space
        }

        # Start generating tune config runs
        start_generate_hp(kwargs=kwargs)
        
        # Retrieve all runs from connect archive
        retrieved_run = run_records.read_all()
        
        # Loop thru all runs with run_id prefixed with "optim" and send them to train queue
        for run in retrieved_run:
            if run['key']['run_id'].startswith('optim'):
                # send each hyperparamer config into the train queue
                start_hp_training(project_id, expt_id, run['key']['run_id'])
                print("<><><>><>OPTIM Training started ---------TUNE INTERFACE")

if __name__=="__main__":

    search_space = {
        'algorithm': 'FedProx',
        'rounds': {"_type": "choice", "_value": [1,2]},
        'epochs': 1,
        'lr': 0.001,
        'weight_decay': 0.0,
        'lr_decay': 0.1,
        'mu': 0.1,
        'l1_lambda': 0.0,
        'l2_lambda': 0.0,
        'optimizer': 'SGD',
        'criterion': 'BCELoss', # BCELoss
        'lr_scheduler': 'CyclicLR',
        'delta': 0.0,
        'patience': 10,
        'seed': 42,
        'is_snn': False,
        'precision_fractional': 5,
        'base_lr': 0.0005,
        'max_lr': 0.005,
    }

    n_samples = 2

    tuner = RayTuneTuner()
    tuner.tune("test_project", "test_experiment", search_space, n_samples)
    #