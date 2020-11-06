from subprocess import Popen
"""
Simple wrapper function for HardwareStatsLogger for starting the logging of hardware stats
"""
import os
from SynergosLogger import syn_logger_config as config

p = None
def run(component, file_path, class_name, function_name):
    """
    args:
        component: Synergos component either TTP or Worker for the HardwareStatsLogger, config.TTP or config.WORKER
        file_path: The location of the file path that call this function
    """
    global p
    HARDWARE_STATS_LOGGER = component['HARDWARE_STATS_LOGGER'] # file path of the hardware stats logger
    p = Popen(['python', HARDWARE_STATS_LOGGER, file_path, class_name, function_name]) # Start the hardware monitoring process

def terminate():
    p.kill() # Sending the SIGTERM signal to the child. Terminate the hardware monitoring process