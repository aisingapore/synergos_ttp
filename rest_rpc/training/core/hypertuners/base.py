#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import logging
import os
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

    def __init__(self, platform: str, log_dir: str = None):
        self.platform = platform
        self.log_dir = log_dir

    ############
    # Checkers #
    ############

    def is_running(self):
        """ Checks if the execution of current tunable is still in progress.
            This function should be overriden in subclass with custom handlers.
        """
        raise NotImplementedError("Please override BaseTuner.is_running() in subclass!")

    ###########    
    # Helpers #
    ###########

    def generate_output_directory(self) -> str:
        """ Generates output directory for logging optimization outputs & 
            caches.

        Returns:
            Destination filepath (str)
        """
        return os.path.join(self.log_dir, self.platform)

    ##################
    # Core Functions #
    ##################

    def tune(self):
        """ Wrapper function that encapsulates all tuning operations. This 
            function should be overriden in subclass with custom handlers.
        """
        raise NotImplementedError("Please override BaseTuner.tune() in subclass!")