from subprocess import Popen
"""
Simple wrapper function for HardwareStatsLogger for starting the logging of hardware stats
"""
import os
from SynergosLogger import config

p = None
def run(file_path, class_name, function_name):
    """
    args:
        file_path: The location of the file path that call this function
    """
    global p
    HARDWARE_STATS_LOGGER = config.HARDWARE_STATS_LOGGER
    p = Popen(['python', HARDWARE_STATS_LOGGER, file_path, class_name, function_name]) # Start the hardware monitoring process

def terminate():
    p.kill() # Sending the SIGTERM signal to the child. Terminate the hardware monitoring process