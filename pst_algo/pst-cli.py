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
import platform
import getopt
import glob
import re
import cv2
import numpy as np
import pandas as pd
import logging
import psutil
import multiprocessing as mp
import algo.blobs as blobs
import algo.kpi as kpi
import config.config as cf

from algo.structs import Frame, Dot

from functools import partial
from datetime import timedelta
from logging.handlers import QueueHandler
from config.logging import setup_logger, logger_process


sys.dont_write_bytecode = True # Disables __pycache__

def pipeline(queue, df_lst, df_frame_lst, frame_nums, maps_xy, maps_dxdy, output_path, params, image_file):
    #------Logging------
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    
    frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
    frame_nums.append(frame_num)
    
    image = cv2.imread(image_file)
    if params['driver'] == 'MODEL':
        pass
    elif params['driver'] == 'FATP': # rotate 180 deg as FATP is mounted upside down
        image = np.rot90(image, 2)  
    
    logger.info('Frame %s : %s Processing started', frame_num, params['driver'])
    height, width, _ = image.shape
    
    if ('1001.tiff' not in image_file) and ('1000.tiff' not in image_file):
        fov_dot = blobs.find_fov(image, params, logger, frame_num, height, width)
        logger.info('Frame %s : FOV dot was found at %s', frame_num, fov_dot.__str__())
   
        # Mask the detected FOV dot
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(image_gray)
        cv2.circle(mask, (int(fov_dot.x), int(fov_dot.y)), int(np.sqrt(fov_dot.size / np.pi) + 7), 255, -1)
        image_gray = cv2.bitwise_and(image_gray, cv2.bitwise_not(mask))
        if params['enable_all_saving']:
            cv2.imwrite(os.path.join(output_path, frame_num+'_no_fov.jpeg'), image_gray, [cv2.IMWRITE_JPEG_QUALITY, 40]) 

        frame = blobs.dot_pattern(image_gray, height, width, params, frame_num, logger)
        blobs.draw_dots(image, [fov_dot, frame.center_dot], os.path.join(output_path, frame_num+'_fov_center_dots.jpeg'), enable=True) # For debugging center and FOV dot detection
        blobs.draw_dots(image, frame.dots, os.path.join(output_path, frame_num+'_dots.jpeg'), enable=params['enable_all_saving']) # For debugging blob detection
        frame.draw_lines_on_image(image, width, height, filepath=os.path.join(output_path, frame_num+'_grouped.jpeg'), enable=params['enable_all_saving'])

        maps_xy[frame_num] = frame.map_xy
        maps_dxdy[frame_num] = frame.map_dxdy

    elif '1000.tiff' in image_file:
        fov_dot = blobs.find_fov(image, params, logger, frame_num, height, width)
        logger.info('Frame %s : FOV dot was found at %s', frame_num, fov_dot.__str__())
        blobs.draw_dots(image, [fov_dot], os.path.join(output_path, frame_num+'_fov_dot.jpeg'), enable=True) # For debugging FOV dot detection
        frame = Frame()

    elif '1001.tiff' in image_file:
        fov_dot = Dot(0, 0, 0)
        frame = blobs.dot_pattern(image, height, width, params, frame_num, logger)
        blobs.draw_dots(image, [frame.center_dot], os.path.join(output_path, frame_num+'_center_dot.jpeg'), enable=True) # For debugging center dot detection
        blobs.draw_dots(image, frame.dots, os.path.join(output_path, frame_num+'_dots.jpeg'), enable=params['enable_all_saving']) # For debugging blob detection
        frame.draw_lines_on_image(image, width, height, filepath=os.path.join(output_path, frame_num+'_grouped.jpeg'), enable=params['enable_all_saving'])

        maps_xy[frame_num] = frame.map_xy
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
                                  'median_dot_size' : frame.med_dot_size, 'median_dot_spacing' : frame.med_dot_dist,
                                  'hor_slope' : frame.hor_slope, 'ver_slope' : frame.ver_slope, 'dxdy_spacing' : params['dxdy_spacing'],
                                  'map_x_shift' : params['map_x_shift'], 'map_y_shift' : params['map_y_shift']}, index=[0])
    
    mini_df = pd.DataFrame({'frame_num' : frame_num, 'x' : x, 'y' : y, 'size' : size, 'xi' : xi, 'yi' : yi})
    
    df_frame_lst.append(mini_df_frame)
    df_lst.append(mini_df)
 
    
if __name__ == '__main__':
    if 'Windows' in platform.system():
        mp.freeze_support()    # For Windows Binary Development
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
    csv_file = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + '_dots.csv')
    csv_file_frame = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + '_frames.csv')
    csv_file_summary = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + '_summary.csv')
    
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
            
    # Add two reference images
    overlay_file = os.path.join(cf.input_path, '1000.tiff')
    no_overlay_file = os.path.join(cf.input_path, '1001.tiff')
    
    if os.path.isfile(overlay_file) and os.path.isfile(no_overlay_file):
        image_files.extend([overlay_file, no_overlay_file])

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

    df.to_csv(csv_file, index=False)
    df_frame.to_csv(csv_file_frame, index=False)
    
    maps_xy_sorted = sorted(maps_xy_dct.items(), key=lambda x: int(x[0]))
    maps_xy = [x[1] for x in maps_xy_sorted]
    maps_dxdy_sorted = sorted(maps_dxdy_dct.items(), key=lambda x: int(x[0]))
    maps_dxdy = [x[1] for x in maps_dxdy_sorted]
    
    kpi.find_outliers(df_frame, width=4024, height=3036)  
    summary_df = df_frame.loc[(df_frame['frame_num'] == '1000') | (df_frame['frame_num'] == '1001')]

    if summary_df.empty:
          middle_frame_index = kpi.find_middle_frame(df_frame)
          summary_df = df_frame.loc[[middle_frame_index]].drop(columns=['dist_center_dot', 'dist_fov_center', 'flag_center_dot_outlier', 'flag_fov_dot_outlier', 'flag_slope_outlier']).reset_index(drop=True)
    else:
        df_frame.drop(df_frame.loc[(df_frame['frame_num'] == '1000') | (df_frame['frame_num'] == '1001')].index, inplace=True)
        middle_frame_index = -1
        summary_df = summary_df.drop(columns=['frame_num', 'dxdy_spacing', 'map_x_shift', 'map_y_shift', 'num_frames_processed', 'dist_center_dot', 'dist_fov_center', 'flag_center_dot_outlier', 
                                              'flag_fov_dot_outlier', 'flag_slope_outlier', 'num_center_dot_outlier', 
                                              'num_fov_dot_outlier', 'num_slope_outlier', 'num_total_outlier']).sum().to_frame().T # Combine both rows into one
        summary_df['frame_num'] = '1000-1001'
        summary_df['dxdy_spacing'] = params['dxdy_spacing']
        summary_df['map_x_shift'] = params['map_x_shift']
        summary_df['map_y_shift'] = params['map_y_shift']
        summary_df['num_frames_processed'] = len(df_frame.index)
        summary_df['num_center_dot_outlier'] = df_frame['num_center_dot_outlier']
        summary_df['num_fov_dot_outlier'] = df_frame['num_fov_dot_outlier']
        summary_df['num_slope_outlier'] = df_frame['num_slope_outlier']
        summary_df['num_total_outlier'] = df_frame['num_total_outlier']

    summary_df = kpi.eval_KPIs(df_frame, params, summary_df, int(middle_frame_index), maps_xy, maps_dxdy)
    summary_df.to_csv(csv_file_summary, index=False)
    
    """ summary_df.to_csv(csv_file_summary, columns=['frame_num', 'total_dots', 'center_dot_x', 'center_dot_y', 'fov_dot_x', 'fov_dot_y', 
                                                 'median_dot_size', 'median_dot_spacing', 'hor_slope', 'ver_slope', 'dxdy_spacing' ,
                                                 'map_x_shift', 'map_y_shift', 'num_frames_processed', 'num_center_dot_outlier', 
                                                 'num_fov_dot_outlier', 'num_slope_outlier', 'num_total_outlier'], index=False) """
    

    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

    
    