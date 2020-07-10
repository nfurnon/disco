import logging
import os
import sys

def set_up_log(logfile='', level=0):
    """Sets up root logger.

    Args:
        logfile (str, optional): Log file. Pass empty string to write to std.err (Default: '')
        level (int, optional): Verbosity level (Default: 0)
            * 0: warnings only
            * 1: info and warnings
            * otherwise: debug, info and warnings

    Returns:
        logging.Logger: Formatted root logger
    """
    log_format = '[%(levelname)s] %(asctime)s %(funcName)s: %(message)s'
    time_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, time_format)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers = [handler]
    if level == 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    return logger
