''' 
Config class to manage the network information of all containers
'''
################################################### Change the following configuration to adhere your needs ###################################################
GRAYLOG_SERVER = "graylog" # change to 127.0.0.1 if testing locally without TTP/Worker or graylog if testing on TTP/Worker
LOGGING_VARIANT = "graylog" # use graylog server or basic logging
SYN_COMPONENT = "ttp" # change to "ttp" if working on ttp container else "worker"
SYSMETRICS_COMPONENT = "sysmetrics"

### default logging port 12201 to 12203 which is to be created in the graylog server
TTP_PORT = 12201
WORKER_PORT = 12202
SYSMETRICS_PORT = 12203

### default path for the hardware stats logger file
HARDWARE_STATS_LOGGER_LOCAL = "/Users/kelvinsoh/Desktop/Synergos/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py"
HARDWARE_STATS_LOGGER_TTP = "/ttp/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py"
HARDWARE_STATS_LOGGER_WORKER = "/worker/synergos_logger/HardwareStatsLogger/HardwareStatsLogger.py"

################################################### Change the following  configuration to adhere your needs ###################################################


COMPONENT_PORT = {"ttp": TTP_PORT, "worker": WORKER_PORT, "sysmetrics": SYSMETRICS_PORT}
COMPONENT_HARDWARE_STATS_LOGGER = {"ttp": HARDWARE_STATS_LOGGER_TTP, "worker": HARDWARE_STATS_LOGGER_WORKER}

# HARDWARE_STATS_LOGGER = "/ttp" # change to ttp if using ttp container else worker
COMPONENT = {"LOGGER": SYN_COMPONENT, "SERVER": GRAYLOG_SERVER, "PORT": COMPONENT_PORT[SYN_COMPONENT], "HARDWARE_STATS_LOGGER": COMPONENT_HARDWARE_STATS_LOGGER[SYN_COMPONENT]}
SYSMETRICS = {"LOGGER": SYSMETRICS_COMPONENT, "SERVER": GRAYLOG_SERVER, "PORT": COMPONENT_PORT[SYSMETRICS_COMPONENT], "HARDWARE_STATS_LOGGER": COMPONENT_HARDWARE_STATS_LOGGER[SYN_COMPONENT]}