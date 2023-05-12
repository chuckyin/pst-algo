#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:30:22 2023

@author: melshaer0612
"""

import sys
import os
import json


sys.dont_write_bytecode = True # Disables __pycache__

current_path = os.getcwd()


def get_version():
    __version__ = '23.05.09p_new_proc'
    
    return __version__      


def config(dataset_folder, params_file):
    global input_path, output_path
    
    input_path = os.path.join(current_path, 'data', dataset_folder, 'input/')
    output_path = os.path.join(current_path, 'data', dataset_folder, 'output/')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Default parameters
    params = {'min_dist' : 3,
              'min_threshold' : 1,
              'max_threshold' : 1000,
              'threshold_step': 2,
              'min_area' : 30,
              'max_area' : 2000,
              'invert' : True,
              'subtract_median': False,
              'filter_byconvexity' : True,
              'min_convexity': 0.6,
              'filter_bycircularity' : False,
              'ratio' : 0.2,
              'num_dots_miss' : 6,
              'accepted_ratio' : 0.3,
              'driver' : 'MODEL',
              'dxdy_spacing' : 4, 
              'filter_size' : 3,
              'binary_threshold' : 100,
              'num_frames' : 10,
              'filter_percent' : 15,
              'model_center_y_shift' : -1.6,
              'fatp_center_y_shift' : 0}
        
    if os.path.isfile(os.path.join(current_path, 'config', params_file)):
        with open(os.path.join(current_path, 'config', params_file)) as pf:
            updated_params = json.load(pf)
            params.update(updated_params)
            
    return params
