''' 
Config class to manage the network information of all containers
'''
GRAYLOG_SERVER = "graylog" # change to 127.0.0.1 if testing locally or graylog if testing on TTP/Worker
TTP_PORT = 12201
WORKER_PORT = 12202
SYSMETRICS_PORT = 12203
# HARDWARE_STATS_LOGGER = "/Users/kelvinsoh/Desktop/Synergos/graylog/HardwareStatsLogger/HardwareStatsLogger.py"
HARDWARE_STATS_LOGGER = "/worker/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py" # change to ttp if using ttp container else worker

TTP = {"LOGGER": "TTP", "SERVER": GRAYLOG_SERVER, "PORT": TTP_PORT}
WORKER = {"LOGGER": "WORKER", "SERVER": GRAYLOG_SERVER, "PORT": WORKER_PORT}
SYSMETRICS = {"LOGGER": "sysmetrics", "SERVER": GRAYLOG_SERVER, "PORT": SYSMETRICS_PORT}