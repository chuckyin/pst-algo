#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:24:13 2023

@author: melshaer0612
"""
print('Importing Dependencies')
import os
import sys
import getopt
import glob
import cv2
import numpy as np
import pandas as pd
import time
import pst_algo.blob_detection as bd

from datetime import timedelta
#from pst_algo import process_csv
print('Finished Importing')

#----------------

def write_log(*args, filepath):
    log_file = open(filepath, 'a+')
    line = ' '.join([str(a) for a in args])
    log_file.write(line + '\n')
    print(line)
    

def main(argv):
    current_path = os.getcwd()
    dataset_folder = ''
    opts, args = getopt.getopt(argv,'hd:',['folder='])
    for opt, arg in opts:
        if opt == '-h':
            print ('exe -d <dataset>')
            sys.exit()
        elif opt in ('-d', '--dataset'):
            dataset_folder = arg
        
    print ('Dataset: ', os.path.join(current_path, 'data', dataset_folder))
    
    start_time = time.monotonic()

    input_path = os.path.join(current_path, 'data', dataset_folder, 'input/')
    output_path = os.path.join(current_path, 'data', dataset_folder, 'output/')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    image_files = glob.glob(input_path + '*.tiff')
    log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.txt'
    csv_file = os.path.join(output_path, time.strftime('%Y%m%d-%H%M%S') + '_indexed_dots.csv')
    
    df = pd.DataFrame()
    cnt = 0
    for image_file in image_files:
        frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
        image = cv2.imread(os.path.join(current_path, image_file))
        print('Frame', frame_num, ': Processing started')
        height, width, _ = image.shape
        
        frame = bd.find_dots(image)
        print('Frame', frame_num, ': Finding dots is complete, found ', str(len(frame.dots)), 'dots')
        
        frame.center_dot = bd.find_center_dot(frame.dots, height, width)
        print('Frame', frame_num, ': Center dot was found at ', frame.center_dot.__str__())
        
        fov_dot = bd.find_fov(image, height, width)
        print('Frame', frame_num, ': Finding fov dot is complete')
        if fov_dot is None:
            write_log('ERROR: Could not find FOV point for frame ', str(frame_num), filepath=os.path.join(output_path, log_file))          
        else:
            bd.draw_dots(image, [fov_dot], filepath=os.path.join(output_path, frame_num+'_fov.tiff')) # For debugging FOV dot detection
        
        bd.draw_dots(image, frame.dots, filepath=os.path.join(output_path, frame_num+'_dots.tiff')) # For debugging blob detection
        
        med_size, med_dist = frame.calc_dot_size_dist()
        print('Dot Size:', med_size, 'Distance:', med_dist)
        
        print('Starting slope calculations for frame', frame_num)
        proc = bd.prep_image(image, normalize_and_filter=True, binarize=False)
        init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = bd.get_initial_slopes(proc, height, width, ratio=0.3)
        hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
        print('HSlope:', hor_slope, 'VSlope:', ver_slope)
        hor_lines, ver_lines = frame.group_lines()
        frame.plot_lines_dots(width, height, filepath=os.path.join(output_path, frame_num+'_grouped.tiff'))
        frame.find_index()
        print('Finished indexing calculations for frame', frame_num)
        
        # Prepare to save results
        x, y, xi, yi, size = [], [], [], [], []
        for k, v in frame.dotsxy_indexed.items():
            x.append(k[0])
            y.append(k[1])
            xi.append(v[0][0])
            yi.append(v[0][1])
        pts = list(zip(x, y))
        
        xpts = [dot.x for dot in frame.dots]
        ypts = [dot.y for dot in frame.dots]
        sizepts = [dot.size for dot in frame.dots]
        l = list(zip(xpts, ypts, sizepts))
        # Only save the indexed dots
        for pt in pts:
            size.append([item[2] for item in l if (item[0] == pt[0]) and (item[1] == pt[1])][0])
        
        # Write results to dataframe and csv
        mini_df = pd.DataFrame({'x' : x, 'y' : y, 'size' : size, 'xi' : xi, 'yi' : yi})
        mini_df['frame_num'] = frame_num
        try:
            mini_df['x_fov'] = fov_dot.x
            mini_df['y_fov'] = fov_dot.y
        except AttributeError:
            mini_df['x_fov'] = np.nan
            mini_df['y_fov'] = np.nan   
        cnt += 1
        df = pd.concat([df, mini_df])
        print('Total frames processed is', str(cnt), '/', len(image_files))
        
    df.to_csv(csv_file)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    #process_csv(csv_file)