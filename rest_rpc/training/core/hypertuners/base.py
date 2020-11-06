#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import logging
import os
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Libs
import psutil
import yaml

# Custom
from rest_rpc import app

##################
# Configurations #
##################

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

src_dir = app.config['SRC_DIR']
cores_used = app.config['CORES_USED']
gpu_count = app.config['GPU_COUNT']

################################################
# Hyperparameter Tuning Base Class - BaseTuner #
################################################

class BaseTuner:
    """
    Base class for performing hyperparameter tuning. This class serves as the
    interfacing layer to communicate & setup runs within different tuning 
    backends that exists outside of Python. 
    
    Supported platforms include:
        1) Microsoft NNI
        2) Ray Tune (pending approval)

    Attributes:
        platform (str): What hyperparameter tuning service to use
        log_dir (str): Directory to export cached log files
    """

    def __init__(
        self, 
        registrations: dict,
        platform: str = "nni", 
        log_dir: str = None
    ):
        self.registrations = registrations
        self.platform = platform
        self.log_dir = log_dir

    ###########    
    # Helpers #
    ###########

    @staticmethod
    def execute_command(self, command):
        """
        """
        completed_process = subprocess.run(
            shlex.split(command),
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        results = completed_process.stdout
        
        return results


    def _generate_search_config(
        self, 
        search_space: Dict[str, Dict[str, Union[str, bool, int, float]]]
    ):
        """ Takes in a user-defined search space for tuning a particular 
            process/function and exports it to a JSON file

        Args:
            search_space (dict): Dictionary containing ranges/options to test
                for 
        """
        search_config_path = os.path.join(
            self.log_dir, 
            self.platform, 
            f"{self.platform}_search_space.yaml"
        )
        with open(search_config_path, 'w') as scp:
            json.dump(search_space, scp)
        
        return search_config_path


    def _generate_tuning_config(
        self,
        author_name: str,
        expt_name: str,
        search_config_path: str,
        tuner: str,
        optimize_mode: str,
        trial_concurrency: int = 1,
        max_exec_duration: str = "1h",
        max_trial_num: int = 10,
        is_remote: bool = True,
        use_annotation: bool = True
    ):
        """ Builds configuration YAML file describing current set of experiments


        """
        configurations = OrderedDict()
        configurations['authorName'] = author_name
        configurations['experimentName'] = expt_name
        configurations['trialConcurrency'] = trial_concurrency
        configurations['maxExecDuration'] = max_exec_duration
        configurations['maxTrialNum'] = max_trial_num
        configurations['trainingServicePlatform'] = "remote" if is_remote else "local"
        configurations['searchSpacePath'] = search_config_path
        configurations['useAnnotation'] = use_annotation
        configurations['tuner'] = {
            'builtinTunerName': tuner,
            'classArgs': {
                'optimize_mode': optimize_mode
            }
        }

        tune_dir = os.path.join(
            src_dir, "rest_rpc", "training", "core", "hypertuners" 
        )
        driver_script_path = os.path.join(tune_dir, "nni_driver_script.py")
        configurations['trial'] = {
            'command': f"python {driver_script_path}",
            'codeDir': tune_dir,
            'gpuNum': gpu_count,
            'cpuNum': cores_used,
            'memoryMB': psutil.virtual_memory().available # remaining system ram
        }

        if self.platform == "nni":
            nni_config_path = os.path.join(
                self.log_dir, 
                self.platform, 
                f"{self.platform}_config.yaml"
            )
            with open(nni_config_path, 'w') as ncp:
                yaml.dump(configurations, ncp)

            return nni_config_path


    def generate_start_command(self, config_path: str):
        """
        """
        if self.platform == "nni":
            init_command = shlex.join(
                ['nnictl', 'create', '--config', config_path]
            )
            return init_command

    ##################
    # Core Functions #
    ##################

    def tune(self):
        """
        """
        

        
        subprocess.run(check=True)