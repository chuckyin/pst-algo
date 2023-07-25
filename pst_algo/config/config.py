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
    __version__ = '07.25.23'
    
    return __version__      


def config(dataset_folder, params_file):
    global input_path, output_path
    
    input_path = os.path.join(current_path, dataset_folder, 'input')
    output_path = os.path.join(current_path, dataset_folder, 'output')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Default parameters
    params = {'min_dist' : 3,                    # Minimum distance in pixels between pixel group centers from several binary (thresholded) images above which they are                                          considered as distinct blobs
              'min_threshold' : 1,               # Minimum value for the threshold function for different blob center detections
              'max_threshold' : 1000,            # Maximum value for the threshold function for different blob center detections
              'threshold_step': 2,               # Step threshold for the threshold function between minimum and maximum for blob center detection
              'min_area' : 30,                   # Minimum area of a blob to be detected. Blobs smaller than this value will be ignored
              'max_area' : 2000,                 # Maximum area of a blob to be detected. Blobs larger than this value will be ignored
              'invert' : True,                   # Enables image intensity inversion before blob detection
              'subtract_median': False,          # Enables median image subtraction before blob detection
              'filter_byconvexity' : True,       # Enables filtering blobs by their convexity
              'min_convexity': 0.6,              # Minimum convexity of a blob to be detected. Blobs with convexity values smaller than this will be ignored
              'filter_bycircularity' : False,    # Enables filtering blobs by their circularity
              'ratio' : 0.2,                     # Distance error ratio for dot-line grouping
              'num_dots_miss' : 6,               # Acceptable number of missed dots during grouping
              'accepted_ratio' : 0.3,            # Accepted ratio of grouped dots
              'driver' : 'MODEL',                # Driver type
              'dxdy_spacing' : 4,                # Spacing interval for derivative calculations
              'binary_threshold' : 100,          # Threshold value for FOV (red circle) detection
              'num_frames' : 10,                 # Number of image frames to process
              'filter_percent' : 15,             # Percentage of data to filter out in post processing KPI calculations
              'kernel_pp_size' : 5,              # Window size for peak-to-peak calculations
              'map_y_shift' : 0,                 # Vertical image shift for Stinson
              'map_x_shift' : 0,                 # Horizontal image shift for Stinson
              'enable_all_saving' : False}       # Enable flag to save all intermediate images for debugging
        
    if os.path.isfile(os.path.join(current_path, 'config', params_file)):
        with open(os.path.join(current_path, 'config', params_file)) as pf:
            updated_params = json.load(pf)
            params.update(updated_params)
            
    return params
