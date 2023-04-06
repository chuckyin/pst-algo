#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pupil Swim Tester (PST) Algorithm Entry Point  

@author: melshaer0612@meta.com

Usage:
    pst-cli.py -h
    pst-cli.py -d <dataset_name> [-p <parameter_file_name>]
    
Options:
    -h              Display usage help message
    -d --dataset    Supply name of dataset to be processed
    -p --params     Supply JSON file containing parameters to be used for processing
    
    
"""

import sys
sys.dont_write_bytecode = True # Disables __pycache__

from algo.__main__ import main



if __name__ == '__main__':
    main(sys.argv[1:])