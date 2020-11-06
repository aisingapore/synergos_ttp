import logging
from SynergosLogger.SynergosLogger import SynergosLogger
from SynergosLogger import config
import os
import datetime

# STEP 1: Get the TTP configuration if worker then change to config.WORKER
CONST_TTP = config.TTP

# STEP 2: Initializing SynergosLogger with the graylog server and port
# file_path = os.path.dirname(os.path.abspath(__file__) + '/' + __file__) # The file path of where the logger is being called
file_path = os.path.abspath(__file__)
syn_logger = SynergosLogger(server=CONST_TTP['SERVER'], port=CONST_TTP['PORT'], logging_level=logging.DEBUG, debugging_fields=True, file_path=file_path, logger_name=CONST_TTP['LOGGER'])

# STEP 3: Initialize Synergos logger (Gelf + Structlog)
logging, logger = syn_logger.initialize_structlog()


# Example of logging called within function in a class
class Test:
    def __init__(self):
        self.test = None
    def evaluate(self):
        logging.info("PHASE 1: CONNECT - Submitting TTP & Participant metadata",
         Class=Test.__name__, level_name="INFO")
  
Test().evaluate()