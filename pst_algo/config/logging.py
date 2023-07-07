#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import sys
import logging
import logging.handlers


sys.dont_write_bytecode = True # Disables __pycache__

logging.getLogger('matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def setup_logger(level=logging.DEBUG, filename=None):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s -- %(levelname)s -- %(filename)s:%(lineno)d (%(funcName)s) -- %(message)s')
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    if not root_logger.hasHandlers():
        root_logger.addHandler(stdout_handler)
    
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return logger


def logger_process(queue, setup_logger, log_file):
    setup_logger(filename=log_file)
        
    while True:
        message = queue.get()
        logger = logging.getLogger(__name__)
        if message is None:
            logger.info('Logger process exiting...')
            break
        logger.handle(message)
        
    