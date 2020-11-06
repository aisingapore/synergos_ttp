import logging
import graypy
import datetime
import time

class UsernameFilter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.username = datetime.datetime.now()

    def filter(self, record):
        record.username = self.username
        return True

my_logger = logging.getLogger('test_logger')
my_logger.setLevel(logging.DEBUG)

handler = graypy.GELFTCPHandler('localhost', 12201)
my_logger.addHandler(handler)

my_logger.addFilter(UsernameFilter())

my_logger.info('Hello Graylog from John.')

time.sleep(5)
my_logger.addFilter(UsernameFilter())
my_logger.info('Hello Graylog from John. 5 secs later')