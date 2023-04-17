#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import os
import logging
import logging.handlers

import config.config as cf


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
        file_handler = logging.FileHandler(os.path.join(cf.output_path, filename))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return logger


