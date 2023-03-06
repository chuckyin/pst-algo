#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:30:22 2023

@author: melshaer0612
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
        
    if os.path.isfile(os.path.join(current_path, 'config', params_file)):
        with open(os.path.join(current_path, 'config', params_file)) as pf:
            params  = json.load(pf)
    else:
        params = {'min_dist' : 3,
              'min_threshold' : 1,
              'max_threshold' : 1000,
              'threshold_step' : 2,
              'min_area' : 50,
              'max_area' : 2000,
              'invert' : True,
              'subtract_median' : False,
              'filter_byconvexity' : True,
              'min_convexity' : 0.75,
              'filter_bycircularity' : False,
              'driver' : 'FATP',
              'dxdy_spacing' : 4 }
        