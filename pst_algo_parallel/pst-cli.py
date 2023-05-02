#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pupil Swim Tester (PST) Algorithm Entry Point   -- Parallel Version. 

Runs on multiple core CPUs for faster processing.        

@author: melshaer0612@meta.com

Usage:
    pst-cli.py -h
    pst-cli.py -d <dataset_name> [-p <parameter_file_name>]
    
Options:
    -h              Display usage help message
    -d --dataset    Supply name of dataset to be processed
    -p --params     Supply JSON file containing parameters to be used for processing
    
    
"""
import time

start_time = time.monotonic()

import os
import sys
import getopt
import glob
import re
import cv2
import numpy as np
import pandas as pd
import csv
import logging
import multiprocessing as mp
import algo.blobs as blobs
import algo.kpi as kpi
import config.config as cf

from functools import partial
from datetime import timedelta
from logging.handlers import QueueHandler
from config.logging import setup_logger, logger_process


sys.dont_write_bytecode = True # Disables __pycache__


def pipeline(queue, df_lst, df_frame_lst, frame_nums, maps_xy, maps_dxdy, output_path, params, image_file):
    #------Logging------
    logger = logging.getLogger(__name__)
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    
    frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
    frame_nums.append(frame_num)
    
    image = cv2.imread(image_file)
    if params['driver'] == 'MODEL':
        pass
    elif params['driver'] == 'FATP': # rotate 180 deg as FATP is mounted upside down
        image = np.rot90(image, 2)  
    
    logger.info('Frame %s : Processing started', frame_num)
    height, width, _ = image.shape
    
    fov_dot = blobs.find_fov(image, params, logger, frame_num, height, width)
    logger.info('Frame %s : FOV dot was found at %s', frame_num, fov_dot.__str__())
   
    # Mask the detected FOV dot
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image_gray)
    cv2.circle(mask, (int(fov_dot.x), int(fov_dot.y)), int(np.sqrt(fov_dot.size / np.pi) + 7), 255, -1)
    image_gray = cv2.bitwise_and(image_gray, cv2.bitwise_not(mask))
    cv2.imwrite(os.path.join(output_path, frame_num+'_no_fov.jpeg'), image_gray, [cv2.IMWRITE_JPEG_QUALITY, 40]) 
   
    frame = blobs.find_dots(image_gray, params)
    logger.info('Frame %s : Finding dots is complete, found %d dots', frame_num, len(frame.dots))
    
    frame.center_dot = blobs.find_center_dot(frame.dots, height, width)
    logger.info('Frame %s : Center dot was found at %s', frame_num, frame.center_dot.__str__())
    
    blobs.draw_dots(image, [fov_dot, frame.center_dot], os.path.join(output_path, frame_num+'_fov_center_dots.jpeg'))
    blobs.draw_dots(image, frame.dots, os.path.join(output_path, frame_num+'_dots.jpeg')) # For debugging blob detection
    
    med_size, med_dist = frame.calc_dot_size_dist()
    logger.info('Frame %s Dot Size: %0.2f Distance: %0.2f', frame_num, med_size, med_dist)
    
    logger.info('Starting slope calculations for frame %s', frame_num)
    proc = blobs.prep_image(image, params, normalize_and_filter=True, binarize=False)
    init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = blobs.get_initial_slopes(proc, height, width, ratio=0.3)
    hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
    logger.info('Frame %s HSlope: %0.2f VSlope: %0.2f', frame_num, hor_slope, ver_slope)
    
    hor_lines, ver_lines = frame.group_lines()
    frame.draw_lines_on_image(image, width, height, filepath=os.path.join(output_path, frame_num+'_grouped.jpeg'))
    frame.find_index(logger, frame_num)
    logger.info('Finished indexing calculations for frame %s', frame_num)
    
    # generate maps
    frame.generate_map_xy(logger, frame_num)
    maps_xy[frame_num] = frame.map_xy
    frame.generate_map_dxdy(params['dxdy_spacing'])  #4x spacing
    maps_dxdy[frame_num] = frame.map_dxdy
    
    # Prepare to save results
    x, y, xi, yi, size = [], [], [], [], []
    xpts = [dot.x for dot in frame.dots]
    ypts = [dot.y for dot in frame.dots]
    sizepts = [dot.size for dot in frame.dots]
    
    for xpt, ypt, sizept in list(zip(xpts, ypts, sizepts)):
        x.append(xpt)
        y.append(ypt)
        size.append(sizept)
        if (xpt, ypt) in frame.dotsxy_indexed:
            xi.append(frame.dotsxy_indexed[(xpt, ypt)][0][0])
            yi.append(frame.dotsxy_indexed[(xpt, ypt)][0][1]) 
        else:
            xi.append(np.nan)
            yi.append(np.nan) 
    
    # Write results to dataframe
    mini_df_frame = pd.DataFrame({'frame_num': frame_num,
                                  'total_dots' : len(frame.dots),
                                  'center_dot_x' : frame.center_dot.x, 'center_dot_y' : frame.center_dot.y,
                                  'fov_dot_x' : fov_dot.x, 'fov_dot_y' : fov_dot.y,
                                  'median_dot_size' : med_size, 'median_dot_spacing' : med_dist,
                                  'hor_slope' : hor_slope, 'ver_slope' : ver_slope,
                                  'dxdy_spacing' : params['dxdy_spacing']}, index=[0])
    
    mini_df = pd.DataFrame({'frame_num' : frame_num, 'x' : x, 'y' : y, 'size' : size, 'xi' : xi, 'yi' : yi})
    
    df_frame_lst.append(mini_df_frame)
    df_lst.append(mini_df)
 
    
if __name__ == '__main__':
    current_path = os.getcwd()
    dataset_folder = ''
    params_file = ''
    opts, args = getopt.getopt(sys.argv[1:],'hd:p:')
    
    for opt, arg in opts:
        if opt == '-h':
            print ('exe -d <dataset> -p <optional_parameter_file>')
            sys.exit()
        elif opt in ('-d', '--dataset'):
            dataset_folder = arg
        elif opt in ('-p', '--params'):
            params_file = arg
        
    print ('Dataset: ', os.path.join(current_path, 'data', dataset_folder))
    print ('Parameters File: ', os.path.join(current_path, 'config', params_file))
    
    params = cf.config(dataset_folder, params_file)
    
    log_file = os.path.join(cf.output_path, 'Log_' + time.strftime('%Y%m%d-%H%M%S') + '.log')
    csv_file = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'dots.csv')
    csv_file_frame = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'frames.csv')
    csv_file_summary = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'summary.csv')
    
    logger = setup_logger(filename=log_file)
        
    image_files_all = glob.glob(cf.input_path + '*.tiff')
    image_files_all.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    if params['num_frames'] == 10:
        frame_num_list = ['15', '60', '70', '75', '80', '85', '90', '95', '105', '150']
        image_files = [image_file for image_file in image_files_all 
                      if ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0] in frame_num_list]
    elif params['num_frames'] == 16:
        frame_num_list = ['15', '30', '45', '60', '65', '70', '75', '80', '85', '90', '95', '100', '105', '120', '135', '150']
        image_files = [image_file for image_file in image_files_all 
                      if ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0] in frame_num_list]
    else:
        image_files = image_files_all[::int(np.ceil(len(image_files_all) / 10))] # only take 10 images
    
    frame_nums = mp.Manager().list()
    maps_xy_dct = mp.Manager().dict()
    maps_dxdy_dct = mp.Manager().dict()
    df_lst = mp.Manager().list()
    df_frame_lst = mp.Manager().list()
    
    queue = mp.Manager().Queue()
    listener = mp.Process(target=logger_process, args=(queue, setup_logger, log_file))
    listener.start()
    
    pool = mp.Pool(processes=mp.cpu_count())

    start = time.perf_counter()
    pipeline_partial = partial(pipeline, queue, df_lst, df_frame_lst, frame_nums, maps_xy_dct, maps_dxdy_dct, cf.output_path, params)
    pool.map(pipeline_partial, image_files)
    print(f'Blob Detection time: {round(time.perf_counter() - start, 2)}')
    df = pd.concat(df_lst, ignore_index=True)
    df_frame = pd.concat(df_frame_lst, ignore_index=True)
    print(frame_nums)
    
    queue.put(None)   
    listener.join()

    df_frame.sort_values(by='frame_num', key=lambda x: x.astype('int'), inplace=True, ignore_index=True)
    df_frame['index'] = np.arange(len(df_frame.index))
    
    maps_xy_sorted = sorted(maps_xy_dct.items(), key=lambda x: int(x[0]))
    maps_xy = [x[1] for x in maps_xy_sorted]
    maps_dxdy_sorted = sorted(maps_dxdy_dct.items(), key=lambda x: int(x[0]))
    maps_dxdy = [x[1] for x in maps_dxdy_sorted]
    
    middle_frame_index = kpi.find_middle_frame(df_frame, width=4024, height=3036)
    summary = kpi.eval_KPIs(df_frame, params, middle_frame_index, frame_nums, maps_xy, maps_dxdy)
    
    with open(csv_file_summary, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, summary.keys())
        w.writeheader()
        w.writerow(summary)      
    df.to_csv(csv_file)
    df_frame.to_csv(csv_file_frame)

    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
    

    
    