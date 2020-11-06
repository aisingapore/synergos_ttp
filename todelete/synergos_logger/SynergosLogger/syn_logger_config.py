''' 
Config class to manage the network information of all containers
'''
GRAYLOG_SERVER = "graylog" # change to 127.0.0.1 if testing locally w/o TTP/Worker or graylog if testing on TTP/Worker
LOGGING_VARIANT = "graylog" # use graylog server or without no server
TTP_PORT = 12201
WORKER_PORT = 12202
SYSMETRICS_PORT = 12203
HARDWARE_STATS_LOGGER_LOCAL = "/Users/kelvinsoh/Desktop/Synergos/graylog/HardwareStatsLogger/HardwareStatsLogger.py"
HARDWARE_STATS_LOGGER_TTP = "/ttp/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py"
HARDWARE_STATS_LOGGER_WORKER = "/worker/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py"

# HARDWARE_STATS_LOGGER = "/ttp" # change to ttp if using ttp container else worker

TTP = {"LOGGER": "TTP", "SERVER": GRAYLOG_SERVER, "PORT": TTP_PORT, "HARDWARE_STATS_LOGGER": HARDWARE_STATS_LOGGER_TTP}
WORKER = {"LOGGER": "WORKER", "SERVER": GRAYLOG_SERVER, "PORT": WORKER_PORT, "HARDWARE_STATS_LOGGER": HARDWARE_STATS_LOGGER_WORKER}
SYSMETRICS = {"LOGGER": "sysmetrics", "SERVER": GRAYLOG_SERVER, "PORT": SYSMETRICS_PORT}