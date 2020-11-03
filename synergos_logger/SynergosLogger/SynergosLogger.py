# from pygelf import GelfTcpHandler
import graypy
import logging
import structlog
from structlog import wrap_logger
import sys
import datetime

class StructlogUtils:
    """
    Structlog utilities class for enhancing the functionality of the default logging
    """
    def __init__(self, censor_keys=[], file_path=""):
        """
        args:
            censor_keys: Censor any logs with the specified keys with the value "*CENSORED*"
            file_path: Retrieve the file location where the logging are called
        """
        self.censor_keys = censor_keys
        self.file_path = file_path

    def censor_logging(self, _, __, event_dict):
        """
        censor log information based on the the keys in the list "censor_keys"
        """
        for key in self.censor_keys:
            k = event_dict.get(key)
            if k:
                event_dict[key] = "*CENSORED*"
        return event_dict

    def get_file_path(self, _, __, event_dict):
        event_dict['file_path'] = self.file_path
        return event_dict

    def add_timestamp(self, _, __, event_dict):
        event_dict["timestamp"] = datetime.datetime.utcnow()
        return event_dict

    def graypy_structlog_processor(self, _, __, event_dict):
        print(event_dict)
        args = (event_dict.get('event', ''),)
        kwargs = {'extra': event_dict}
        ## The following are default graypy metrics which is logged to the graylog server
        ## comment these out if the following metrics are necessary
        ## alternatively we can set debugging_fields=False in GELF handler which is
        ## similar to the below kwargs setting
        kwargs['extra']['pid'] = ""
        kwargs['extra']['process_name'] = ""
        kwargs['extra']['thread_name'] = ""
        kwargs['extra']['file'] = ""
        # kwargs['extra']['function'] = ""

        return args, kwargs

class SynergosLogger:
    
    def __init__(self, server, port, logging_level, debugging_fields=False, file_path="", logger_name="std_log", filter_function=[]):
        """
        Initialize configuration for setting up a logging server using structlog
        args:
            file_path: The path of the file with the logging message (Arbitrary Graylog arguments)
            server: The host address of the logging server e.g. 127.0.0.1 for Graylog
            port: The port of the logging server e.g. 9000 for graylog
            logging_level: logging.DEBUG, logging.INFO, logging.WARNING etc..
            logger_name: Identifying different logger by name e.g. TTP, worker_1, worker_2
        """
        self.file_path = file_path
        self.server = server
        self.port = port
        self.logging_level = logging_level
        self.debugging_fields = debugging_fields
        self.logger_name = logger_name
        self.filter_function = filter_function

    
    def add_filter_function(self, logger, filter_function):
        '''
        filter_function: list of filter function (fields) to store in graylog server
        '''
        for i in filter_function:
            logger.addFilter(i)

    def initialize_structlog(self, censor_keys=[], **kwargs):
        """
        Initialize configuration for structlog and pygelf for logging the following messages e.g.
        logging a debug message "hello" to graylog server
        >>> logger.debug("hello")
        {"event": "hello", "logger": "stdlog", "level": "debug", "timestamp": "2020-10-21T05:09:10.868747Z"}
        return:
            syn_logger: A structlog + Pygelf logger
            logger: base logger class
        """
        structlogUtils = StructlogUtils(censor_keys=censor_keys, file_path=self.file_path)
        censor_logging = structlogUtils.censor_logging
        get_file_path = structlogUtils.get_file_path
        add_timestamp = structlogUtils.add_timestamp
        graypy_structlog_processor = structlogUtils.graypy_structlog_processor

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name, # adding logger name
                structlog.stdlib.add_log_level, # addinng logger level
                # structlog.stdlib.add_log_level_number,
                structlog.stdlib.PositionalArgumentsFormatter(),
                censor_logging, # censoring sensitive log messages
                get_file_path,
                # add_timestamp,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
                structlog.processors.StackInfoRenderer(), 
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                graypy_structlog_processor,
                # structlog.stdlib.render_to_log_kwargs,
                # structlog.processors.JSONRenderer(indent=1), # for dictionary in message
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=self.logging_level # default
        )

        '''
        logging handler to connect to graylog server
        Add arbitrary arguments preceded by '_' to the graylog server by specifying more arguments in GelfTCPHandler,
        e.g. To add a new key "test" with value "testing" 
        GelfTcpHandler(host=self.server, port=self.port, _file_path=self.file_path, _test="testing" ...)
        '''
        logger = logging.getLogger(self.logger_name) # logger name must tally for basic logging and structlog
        # logger.setLevel('INFO')
        # handler = graypy.GELFTCPHandler(host=self.server, port=self.port, include_extra_fields=True, _file_path=self.file_path, **kwargs)
        handler = graypy.GELFTCPHandler(host=self.server, port=self.port,
             debugging_fields=self.debugging_fields, facility="", level_names=True) # disable default debugging fields "function", "pid", "process_name", "thread_name"

        if len(self.filter_function) > 0:
            self.add_filter_function(logger, self.filter_function)    
        logger.addHandler(handler)

        syn_logger = structlog.get_logger(self.logger_name)
        return syn_logger, logger