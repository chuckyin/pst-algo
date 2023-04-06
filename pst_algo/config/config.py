#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import os
import json

current_path = os.getcwd()

def config(dataset_folder, params_file):
    global params, input_path, output_path
    
    input_path = os.path.join(current_path, 'data', dataset_folder, 'input/')
    output_path = os.path.join(current_path, 'data', dataset_folder, 'output/')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
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
              'binary_threshold' : 50 }
        
    if os.path.isfile(os.path.join(current_path, 'config', params_file)):
        with open(os.path.join(current_path, 'config', params_file)) as pf:
            updated_params = json.load(pf)
            params.update(updated_params)

        