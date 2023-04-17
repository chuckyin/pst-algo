#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""
import time

start_time = time.monotonic()

import os
import sys
import getopt
import glob
import re
import cv2
import csv
import numpy as np
import pandas as pd
import algo.blobs as blobs
import algo.kpi as kpi
import config.config as cf

from datetime import timedelta
from config.logging import setup_logger


def main(argv):
    current_path = os.getcwd()
    dataset_folder = ''
    params_file = ''
    opts, args = getopt.getopt(argv,'hd:p:')
    
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
    
    cf.config(dataset_folder, params_file)
    
    log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.log'
    csv_file = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'dots.csv')
    csv_file_frame = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'frames.csv')
    csv_file_summary = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'summary.csv')
    
    logger = setup_logger(filename=log_file)
        
    image_files_all = glob.glob(cf.input_path + '*.tiff')
    image_files_all.sort(key=lambda f:int(re.sub('\D', '', f)))
    
    if cf.params['num_frames'] == 10:
        frame_num_list = ['15', '60', '70', '75', '80', '85', '90', '95', '105', '150']
        image_files = [image_file for image_file in image_files_all 
                       if ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0] in frame_num_list]
    elif cf.params['num_frames'] == 16:
        frame_num_list = ['15', '30', '45', '60', '65', '70', '75', '80', '85', '90', '95', '100', '105', '120', '135', '150']
        image_files = [image_file for image_file in image_files_all 
                       if ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0] in frame_num_list]
    else:
        image_files = image_files_all[::int(np.ceil(len(image_files_all) / 10))] # only take 10 images
    
    df = pd.DataFrame() # raw data, 1 blob per line
    df_frame = pd.DataFrame() # frame summary data, 1 frame per line
    
    num = 0
    maps_xy = []
    maps_dxdy = []
    frame_nums = []
    
    # START frame loop
    for image_file in image_files:
        frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
        frame_nums.append(frame_num)
        
        image = cv2.imread(os.path.join(current_path, image_file))
        if cf.params['driver'] == 'MODEL':
            pass
        elif cf.params['driver'] == 'FATP': # rotate 180 deg as FATP is mounted upside down
            image =np.rot90(image, 2)  
        
        logger.info('Frame %s : Processing started', frame_num)
        height, width, _ = image.shape
        
        fov_dot = blobs.find_fov(image, height, width)
        logger.info('Frame %s : FOV dot was found at %s', frame_num, fov_dot.__str__())
        
        # Mask the detected FOV dot
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(image_gray)
        cv2.circle(mask, (int(fov_dot.x), int(fov_dot.y)), int(np.sqrt(fov_dot.size / np.pi) + 7), 255, -1)
        image_gray = cv2.bitwise_and(image_gray, cv2.bitwise_not(mask))
        cv2.imwrite(os.path.join(cf.output_path, frame_num+'_no_fov.jpeg'), image_gray, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        frame = blobs.find_dots(image_gray)
        logger.info('Frame %s : Finding dots is complete, found %d dots', frame_num, len(frame.dots))
        
        frame.center_dot = blobs.find_center_dot(frame.dots, height, width)
        logger.info('Frame %s : Center dot was found at %s', frame_num, frame.center_dot.__str__())
        
        blobs.draw_dots(image, [fov_dot, frame.center_dot], filename=frame_num+'_fov_center_dots.jpeg')
        blobs.draw_dots(image, frame.dots, filename=frame_num+'_dots.jpeg') # For debugging blob detection
        
        med_size, med_dist = frame.calc_dot_size_dist()
        logger.info('Dot Size: %0.2f Distance: %0.2f', med_size, med_dist)
        
        logger.info('Starting slope calculations for frame %s', frame_num)
        proc = blobs.prep_image(image, normalize_and_filter=True, binarize=False)
        init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = blobs.get_initial_slopes(proc, height, width, ratio=0.3)
        hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
        logger.info('HSlope: %0.2f VSlope: %0.2f', hor_slope, ver_slope)
        
        hor_lines, ver_lines = frame.group_lines()
        frame.draw_lines_on_image(image, width, height, filename=frame_num+'_grouped.jpeg')
        frame.find_index()
        logger.info('Finished indexing calculations for frame %s', frame_num)
            
        # Generate maps for current frame
        frame.generate_map_xy()
        maps_xy.append(frame.map_xy)
        frame.generate_map_dxdy(cf.params['dxdy_spacing'])  #4x spacing
        maps_dxdy.append(frame.map_dxdy)
        
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
        mini_df_frame = pd.DataFrame({'frame_num': frame_num, 'index': num,
                                      'total_dots' : len(frame.dots),
                                      'center_dot_x' : frame.center_dot.x, 'center_dot_y' : frame.center_dot.y,
                                      'fov_dot_x' : fov_dot.x, 'fov_dot_y' : fov_dot.y,
                                      'median_dot_size' : med_size, 'median_dot_spacing' : med_dist,
                                      'hor_slope' : hor_slope, 'ver_slope' : ver_slope,
                                      'dxdy_spacing' : cf.params['dxdy_spacing']}, index=[0])
        
        mini_df = pd.DataFrame({'frame_num' : frame_num, 'x' : x, 'y' : y, 'size' : size, 'xi' : xi, 'yi' : yi})
        
        df = pd.concat([df, mini_df])
        df_frame = pd.concat([df_frame, mini_df_frame])
        
        num += 1
        logger.info('Total frames processed is %d / %d', num, len(frame_num_list))
    # END frame loop    
    
    middle_frame_index = kpi.find_middle_frame(df_frame, width, height)
    summary = kpi.eval_KPIs(df_frame, middle_frame_index, frame_nums, maps_xy, maps_dxdy)
    
    with open(csv_file_summary, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, summary.keys())
        w.writeheader()
        w.writerow(summary)      
    df.to_csv(csv_file)
    df_frame.to_csv(csv_file_frame)
    
    logger.handlers.clear()
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

    

    
    
    
    
   
