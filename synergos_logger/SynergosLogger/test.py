import logging
from SynergosLogger import SynergosLogger
# from Constant_Class import Constant_Class
import config
import os
import datetime

'''
Filter function is necessary if we want to log "DYNAMIC" metrics to the graylog server 
for real time monitoring
'''
class TestFilter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.testing_filter = datetime.datetime.now()

    def filter(self, record):
        record.testing_filter = self.testing_filter
        return True

class TestFilter_1(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.testing_filter_1 = datetime.datetime.now()

    def filter(self, record):
        record.testing_filter_1 = self.testing_filter_1
        return True


# initializing TTP container information such as "Server", "Port" and "Logger name"
CONST_TTP = config.TTP

# Initializing SynergosLogger with the graylog server and port
file_path = os.path.dirname(os.path.abspath(__file__) + '/' + __file__) # The file path of where the logger is being called
syn_logger = SynergosLogger(server=CONST_TTP['SERVER'], port=CONST_TTP['PORT'], logging_level=logging.DEBUG, debugging_fields=True, file_path=file_path, logger_name=CONST_TTP['LOGGER'], filter_function=[TestFilter(), TestFilter_1()])

# If there are logs to be censored, pass in optional argument "censor_keys" to censor log messages else empty [].
logging, logger = syn_logger.initialize_structlog(censor_keys=["passwordx", "other_stuff0"])

# logging in structlog format
# logging.info("test", other_stuff1="other_stuff1", other_stuff2=[1,2,3], other_stuff3={"x": [1,2,3]}, other_stuff0=3.1, passwordx='SECRET')


# Retrieving class and function name
class Test:
    def __init__(self):
        self.test = None
    def some_func(self, x, y):
        return x + y

    def something():
        syn_logger.add_filter_function(logger, [TestFilter(), TestFilter_1()])
        logging.critical(event="PHASE 1: CONNECT - Submitting TTP & Participant metadata",
                more_message="Experiment 1 does not exist in project test_project", passwordx="SECRET", other_stuff0="other_stuff0", Class=Test.__name__)
                # ,classes=Test.__name__, function=Test.some_func.__name__, passwordx="SECRET")
Test.something()
# import time
# time.sleep(3)

