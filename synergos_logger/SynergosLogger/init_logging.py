from SynergosLogger.SynergosLogger import SynergosLogger
from SynergosLogger import syn_logger_config as config
# from SynergosLogger import SynergosLogger # test local
# import syn_logger_config as config
import os
import datetime
import logging

# initializing TTP container information such as "Server", "Port" and "Logger name"
CONST_COMPONENT = config.COMPONENT
LOGGING_VARIANT = config.LOGGING_VARIANT

# Initializing SynergosLogger with the graylog server and port
# file_path = os.path.dirname(os.path.abspath(__file__)) # The file path of where the logger is being called
file_path = os.path.abspath(__file__)
syn_logger = SynergosLogger(server=CONST_COMPONENT['SERVER'], port=CONST_COMPONENT['PORT'], logging_level=logging.DEBUG, debugging_fields=True, logger_name=CONST_COMPONENT['LOGGER'], logging_variant=LOGGING_VARIANT)

# If there are logs to be censored, pass in optional argument "censor_keys" to censor sensitive log messages e.g. sys_logger.initialize_structlog(censor_keys=["password", "username"])
logging, logger = syn_logger.initialize_structlog()