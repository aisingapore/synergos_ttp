# https://github.com/tarekziade/system-metrics/blob/master/sysmetrics/__init__.py
import sys
import psutil
import asyncio
import functools
import signal
import logging
import structlog
import json
import os

from SynergosLogger.SynergosLogger import SynergosLogger
from SynergosLogger import syn_logger_config as config

class CPU_Filter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.cpu_percent = psutil.cpu_percent(interval=None)

    def filter(self, record):
        record.cpu_percent = self.cpu_percent
        return True

class Memory_Filter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.memory_total = psutil.virtual_memory().total
        self.memory_available = psutil.virtual_memory().available
        self.memory_used = psutil.virtual_memory().used
        self.memory_free = psutil.virtual_memory().free

    def filter(self, record):
        record.memory_total = self.memory_total
        record.memory_available = self.memory_available
        record.memory_used = self.memory_used
        record.memory_free = self.memory_free
        return True

class DiskIO_Filter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.disk_read_counter = psutil.disk_io_counters().read_count
        self.disk_write_count = psutil.disk_io_counters().write_count
        self.disk_read_bytes = psutil.disk_io_counters().read_bytes
        self.disk_write_bytes = psutil.disk_io_counters().write_bytes

    def filter(self, record):
        record.disk_read_counter = self.disk_read_counter
        record.disk_write_count = self.disk_write_count
        record.disk_read_bytes = self.disk_read_bytes
        record.disk_write_bytes = self.disk_write_bytes
        return True

class NetIO_Filter(logging.Filter):
    def __init__(self):
        # In an actual use case would dynamically get this
        # (e.g. from memcache)
        self.net_bytes_sent = psutil.net_io_counters().bytes_sent
        self.net_bytes_recv = psutil.net_io_counters().bytes_recv
        self.net_packets_sent = psutil.net_io_counters().packets_sent
        self.net_packets_recv = psutil.net_io_counters().packets_recv

    def filter(self, record):
        record.net_bytes_sent = self.net_bytes_sent
        record.net_bytes_recv = self.net_bytes_recv
        record.net_packets_sent = self.net_packets_sent
        record.net_packets_recv = self.net_packets_recv
        return True

loop = asyncio.get_event_loop()
RESOLUTION = 1. # How often to probe for hardware usage in seconds

# initializing TTP container information such as "Server", "Port" and "Logger name"
CONST_SYSMETRICS = config.SYSMETRICS

# Initializing SynergosLogger with the graylog server and port
# file_path = os.path.dirname(os.path.abspath(__file__) + '/' + __file__)
file_path = sys.argv[1]
class_name = sys.argv[2]
function_name = sys.argv[3]

syn_logger = SynergosLogger(server=CONST_SYSMETRICS['SERVER'], port=CONST_SYSMETRICS['PORT'], logging_level=logging.DEBUG, file_path=file_path, logger_name=CONST_SYSMETRICS['LOGGER'], filter_function=[CPU_Filter(), Memory_Filter(), DiskIO_Filter(), NetIO_Filter()])
# If there are logs to be censored, pass in optional argument "censor_keys" to censor log messages.
logging, logger = syn_logger.initialize_structlog()

def _exit(signame):
    loop.stop()

def _probe():
    syn_logger.add_filter_function(logger, [CPU_Filter(), Memory_Filter(), DiskIO_Filter(), NetIO_Filter()])
    # info = {'cpu_percent': psutil.cpu_percent(interval=None)}
    logging.info("logging cpu & memory usage", file_path=file_path, Class=class_name, function=function_name)
    # logging.info(json.dumps(info))
    loop.call_later(RESOLUTION, _probe)
    

if __name__ == '__main__':
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(getattr(signal, signame),
                                functools.partial(_exit, signame))
    # loop.add_signal_handler(signal.SIGINT, _exit)
    # loop.add_signal_handler(signal.SIGTERM, _exit)
    loop.call_later(RESOLUTION, _probe)
    print("Probing cpu, memory and disk...")
    try:
        loop.run_forever()
        # loop.run_until_complete(test())
    finally:
        loop.close()

# Graylog search debugging,
# query = search from message in structlog
# graylog fields for populating data in dashboard for real time monitoring
# timestamp:["2019-07-23 09:53:08.175" TO "2019-07-23 09:53:08.575"]
# timestamp:["2020-10-29 06:20:14.940" TO "2020-10-29 06:22:14.940"]