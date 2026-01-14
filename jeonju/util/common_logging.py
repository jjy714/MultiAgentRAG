import os
import logging

formatter = logging.Formatter('%(asctime)s:%(module)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S')

class log :
    def console_log(logger) : 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    def file_log(logger, filepath) :
        timedfilehandler_info = logging.handlers.TimedRotatingFileHandler(filename='./log/' + filepath, when='midnight', interval=1, encoding='utf-8')
        timedfilehandler_info.setFormatter(formatter)
        timedfilehandler_info.suffix = "%Y%m%d"
        timedfilehandler_info.setLevel(logging.INFO)
        logger.addHandler(timedfilehandler_info)
    
    def error_log(logger, filepath) :
        timedfilehandler_error = logging.handlers.TimedRotatingFileHandler(filename='./log/' + filepath, when='midnight', interval=1, encoding='utf-8')
        timedfilehandler_error.setFormatter(formatter)
        timedfilehandler_error.suffix = "%Y%m%d"
        timedfilehandler_error.setLevel(logging.ERROR)
        logger.addHandler(timedfilehandler_error)