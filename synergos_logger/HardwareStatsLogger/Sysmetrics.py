#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from subprocess import Popen

# Custom
from SynergosLogger import syn_logger_config as config

"""
Simple wrapper function for HardwareStatsLogger for start the logging for hardware stats
"""

def run(hardware_stats_logger, file_path, class_name, function_name):
    """
    args:
        component: Synergos component either TTP or Worker for the HardwareStatsLogger, config.TTP or config.WORKER
        file_path: The location of the file path that call this function
    """
    process = Popen(['python', hardware_stats_logger, file_path, class_name, function_name]) # Start the hardware monitoring process
    return process

def terminate(process):
    process.kill() # Sending the SIGTERM signal to the child. Terminate the hardware monitoring process