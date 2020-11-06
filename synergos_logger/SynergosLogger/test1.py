# from SynergosLogger.init_logging import logging
import init_logging
logging = init_logging.logging
logging.info('x')

# Synergos logging
# from SynergosLogger import init_logging
# logging = init_logging.logging

# logging.info("x")
# # import test

# import logging
# from SynergosLogger import SynergosLogger
# # from Constant_Class import Constant_Class
# import config
# import os
# import datetime


# # initializing TTP container information such as "Server", "Port" and "Logger name"
# CONST_TTP = config.TTP

# # Initializing SynergosLogger with the graylog server and port
# # file_path = os.path.dirname(os.path.abspath(__file__)) # The file path of where the logger is being called
# file_path = os.path.abspath(__file__)
# syn_logger = SynergosLogger(server=CONST_TTP['SERVER'], port=12202, logging_level=logging.DEBUG, debugging_fields=True, file_path=file_path, logger_name=CONST_TTP['LOGGER'])

# # If there are logs to be censored, pass in optional argument "censor_keys" to censor log messages else empty [].
# logging, logger = syn_logger.initialize_structlog()
# logging.info("x")


# # Retrieving class and function name
# # class Test:
# #     def __init__(self):
# #         self.test = None
# #     def some_func(self, x, y):
# #         return x + y

# #     def something():
# #         syn_logger.add_filter_function(logger, [TestFilter(), TestFilter_1()])
# #         logging.critical(event="PHASE 1..: CONNECT - Submitting TTP & Participant metadata",
# #                 more_message="Experiment 1 does not exist in project test_project", passwordx="SECRET", other_stuff0="other_stuff0", Class=Test.__name__)
# #                 # ,classes=Test.__name__, function=Test.some_func.__name__, passwordx="SECRET")
# # Test.something()
# # import time
# # time.sleep(3)

