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
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import pst_algo.blob_detection as bd
import config.config as cf

from datetime import timedelta
#from pst_algo import process_csv
print('Finished Importing Dependencies')

#----------------
def shift_fillnan (arr, shift, axis):
    #shift 2d array with nan fillin
    arr = np.roll(arr, shift, axis)
    if axis == 0:
        if shift >= 0:
            arr[:shift, :] =np.nan
        else:
            arr[shift:, :] =np.nan
    elif axis == 1:
        if shift >= 0:
            arr[:, :shift] =np.nan
        else:    
            arr[:, shift:] =np.nan
    
    return arr    


def calc_median_map(maps, min_num=3):
    #calculate the median map of a maps (list), with min number of valid elements (will output nan if <min_num)
    cnt = 0
    maps_arr = np.asarray(maps)
    l, xi, yi, dim = np.shape(maps_arr)
    output = np.empty((xi, yi, dim))
    if min_num > l:
        min_num = l
        print ("Warning: min number of valid elements reduce to: ", str(l))
    
    for i in range(xi):
        for j in range(yi):
            for k in range(dim):
                if np.count_nonzero(~np.isnan(maps_arr[:, i, j, k])) >= min_num:
                    output[i, j, k] = np.nanmedian(maps_arr[:, i, j, k])
                    cnt += 1
                else:
                    output[i,j,k] = np.nan
    print ('Median map calculation complete with' , str(cnt), 'dots' )                
    return output


def calc_distance(map_xy, ref0):
    #calculate distance map, relative to ref0, ref0 could be map of same shape, or x,y coor 1-dim array
    xi_full, yi_full, _ = map_xy.shape

    map_distance = np.empty((xi_full, yi_full))
    map_distance[:,:] = np.nan
    
    map_xy_delta = map_xy - ref0
    
    for i in range(xi_full):
        for j in range(yi_full):
            map_distance[i,j] = np.sqrt((map_xy_delta[i, j, 0] ** 2 + map_xy_delta[i, j, 1] ** 2))
    
    return map_distance


def plot_map_norm(map_norm, fname_str=''):
    xi_full, yi_full, dim = map_norm.shape
    xi_range = int((xi_full - 1) / 2)
    yi_range = int((yi_full - 1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range,xi_range,xi_full),np.linspace(-yi_range,yi_range,yi_full))
    #map_norm = np.ma.array(map_norm, mask=np.isnan(map_norm))
    
    for i in [0,1]: #x,y
        if i == 0:
            axis = 'x'
        else:
            axis = 'y'
        fig = plt.figure(figsize=(5, 5),dpi=200) 
        ax = fig.add_subplot(111)
        plt.title('Normalized' + axis + '-Spacing Change')
        plt.xlabel('xi')
        plt.ylabel('yi')
        levels = np.linspace(0.95, 1.05, 11)
        plt.contourf(X, Y, map_norm[:,:,i].T, levels=levels, cmap='seismic', extend='both')  #cmap ='coolwarm','bwr'
        ax.set_aspect('equal')
    
        plt.xlim(-xi_range, xi_range)
        plt.ylim(yi_range, -yi_range)
        
        radii = [25, 45]
    
        for radius in radii:    
            theta = np.linspace(0, 2 * np.pi, 100)
            a = radius * np.cos(theta)
            b = radius * np.sin(theta)
            plt.plot(a, b, color='green', linestyle=(0, (5, 5)), linewidth=0.4)  #'loosely dotted'
            
        plt.grid(color='grey', linestyle='dashed', linewidth=0.2)
        plt.rcParams['font.size'] = '5'
        plt.colorbar()
        plt.savefig(os.path.join(cf.output_path, 'Normalized_Map_' + fname_str + axis +'.png'))

    
def plot_map_global(map_global, fname_str=''):   
    xi_full, yi_full = map_global.shape
    xi_range = int((xi_full -1) / 2)
    yi_range = int((yi_full -1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full), np.linspace(-yi_range, yi_range, yi_full))
    #map_global = np.ma.array(map_global, mask=np.isnan(map_global))

    fig = plt.figure(figsize=(5, 5), dpi=200) 
    ax = fig.add_subplot(111)
    plt.title('Global PS Map' + fname_str + '[degree]')
    plt.xlabel('xi')
    plt.ylabel('yi')
    levels = np.linspace(0, 1.0, 6)
    plt.contourf(X, Y, map_global[:, :].T, levels=levels, cmap='coolwarm', extend ='max')  #cmap ='coolwarm','bwr'
    ax.set_aspect('equal')

    plt.xlim(-xi_range, xi_range)
    plt.ylim(yi_range, -yi_range)
    
    radii = [25,45]

    for radius in radii:    
        theta = np.linspace(0, 2 * np.pi, 100)
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        plt.plot(a, b, color='green', linestyle=(0, (5, 5)), linewidth=0.4)  #'loosely dotted'
        
    plt.grid(color='grey', linestyle='dashed', linewidth=0.2)
    plt.rcParams['font.size'] = '5'
    plt.colorbar()
    plt.savefig(os.path.join(cf.output_path, 'Global_PS_Map_' + fname_str + '.png'))       

    
def offset_map_fov (map_input, xi_fov, yi_fov):
    #offset map and create FOV map (center of display)
    map_output = shift_fillnan(map_input, int(np.round(-xi_fov)), axis=0) #shift map relative to FOV 
    map_output = shift_fillnan(map_output, int(np.round(-yi_fov)), axis=1)
    
    return map_output

    
def calc_parametrics_local (map_local, map_fov):
    #calculate the parametrics based on the map
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    axis = ['x', 'y']
    th = [1, 5]  #in percent
    
    #define zones
    for i in range(len(radii)):
        zone = (map_fov > start_r) * (map_fov <= radii[i])
        start_r = radii[i]        
        mapp_both_x = []
        for j in range(len(axis)):            
            mapp = map_local[:, :, j]
            summary['local_areatotal_d' + axis[j] + '_' + str(radii[i])] = np.count_nonzero(~np.isnan(mapp[zone]))
            summary['local_max_d' + axis[j] + '_' + str(radii[i])] = (np.nanmax(mapp[zone]) - 1) * 100
            summary['local_min_d' + axis[j] + '_' + str(radii[i])] = (np.nanmin(mapp[zone]) - 1) * 100
            summary['local_pct99_d' + axis[j] + '_' + str(radii[i])] = (np.nanpercentile(mapp[zone], 99) - 1) * 100
            summary['local_pct1_d' + axis[j] + '_'+ str(radii[i])] = (np.nanpercentile(mapp[zone], 1) - 1) * 100
            summary['local_rms_d' + axis[j] + '_' + str(radii[i])] = (np.nanstd(mapp[zone])) * 100
            for k in range(len(th)): 
                mapp_pos = (mapp >= 1 + th[k] / 100) * zone
                mapp_neg = (mapp <= 1 - th[k] / 100) * zone
                mapp_both = ((mapp >= 1 + th[k]/100) + (mapp <= 1 - th[k] / 100)) * zone
                summary['local_area_d' + axis[j] + '_th'+ str(th[k]) + 'pctpos_' + str(radii[i])] = np.count_nonzero(mapp_pos)
                summary['local_area_d' + axis[j] + '_th'+ str(th[k]) + 'pctneg_' + str(radii[i])] = np.count_nonzero(mapp_neg)
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pct_' + str(radii[i])] = np.count_nonzero(mapp_both)
                if j == 0: # x maps
                    mapp_both_x.append(mapp_both)
                elif j == 1: # y maps
                    mapp_both_combine = mapp_both + mapp_both_x[k]
                    summary['local_area_combined_th' + str(th[k]) + 'pct_' + str(radii[i])] = np.count_nonzero(mapp_both_combine)
                
    return summary


def calc_parametrics_global (map_global, map_fov, label, frame_num):
    #calculate the parametrics based on the map    
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    #axis = ['x','y']
    #th = [1,5]  #in percent
    summary['global_'+label+'_frame_num']= frame_num
    
    #define zones
    for i in range(len(radii)):
        zone =  (map_fov > start_r) * (map_fov <= radii[i])
        start_r = radii[i]
        summary['global_' + label + '_max_' + str(radii[i])]= np.nanmax(map_global[zone])
        summary['global_' + label + '_pct99_' + str(radii[i])]= np.nanpercentile(map_global[zone], 99)
        summary['global_' + label + '_median_' + str(radii[i])]= np.nanpercentile(map_global[zone], 50)

    return summary


def write_log(*args, filepath):
    log_file = open(filepath, 'a+')
    line = ' '.join([str(a) for a in args])
    log_file.write(line + '\n')
    print(line)
    

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
    
    start_time = time.monotonic()
    
    cf.config(dataset_folder, params_file)
        
    image_files = glob.glob(cf.input_path + '*.tiff')
    image_files.sort(key=lambda f:int(re.sub('\D', '', f)))
    log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.txt'
    csv_file = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'dots.csv')
    csv_file_frame = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'frames.csv')
    csv_file_summary = os.path.join(cf.output_path, time.strftime('%Y%m%d-%H%M%S') + 'summary.csv')
    
    df = pd.DataFrame() #raw data, 1 blob per line
    df_frame = pd.DataFrame() #frame summary data, 1 frame per line
    cnt = 0
    maps_xy = []
    maps_dxdy = []
    frame_nums = []
    for image_file in image_files:
        frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
        frame_nums.append(frame_num)
        df_frame_mini = pd.DataFrame({'frame_num':[frame_num],'index':[cnt]})
        image = cv2.imread(os.path.join(current_path, image_file))
        
        if cf.params['driver'] == 'MODEL':
            image = np.fliplr(image)
        elif cf.params['driver'] == 'FATP': # rotate 180 deg as FATP is mounted upside down
            image =np.rot90(image, 2)  
        
        print('Frame', frame_num, ': Processing started')
        height, width, _ = image.shape
        
        frame = bd.find_dots(image)
        print('Frame', frame_num, ': Finding dots is complete, found ', str(len(frame.dots)), 'dots')
        df_frame_mini['total_dots'] = len(frame.dots)
        
        frame.center_dot = bd.find_center_dot(frame.dots, height, width)
        print('Frame', frame_num, ': Center dot was found at ', frame.center_dot.__str__())
        df_frame_mini['center_dot_x'] = frame.center_dot.x
        df_frame_mini['center_dot_y'] = frame.center_dot.y
        
        fov_dot = bd.find_fov(image, height, width)
        print('Frame', frame_num, ': Finding fov dot is complete')
        if fov_dot is None:
            write_log('ERROR: Could not find FOV point for frame ', str(frame_num), filename=log_file)          
        else:
            bd.draw_dots(image, [fov_dot], filename=frame_num+'_fov.jpeg') # For debugging FOV dot detection
        
        bd.draw_dots(image, frame.dots, filename=frame_num+'_dots.jpeg') # For debugging blob detection
        df_frame_mini['fov_dot_x'] = fov_dot.x
        df_frame_mini['fov_dot_y'] = fov_dot.y  
        
        med_size, med_dist = frame.calc_dot_size_dist()
        print('Dot Size:', med_size, 'Distance:', med_dist)
        df_frame_mini['median_dot_size'] = med_size
        df_frame_mini['median_dot_spacing'] = med_dist
        
        print('Starting slope calculations for frame', frame_num)
        proc = bd.prep_image(image, normalize_and_filter=True, binarize=False)
        init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = bd.get_initial_slopes(proc, height, width, ratio=0.3)
        hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
        print('HSlope:', hor_slope, 'VSlope:', ver_slope)
        df_frame_mini['hor_slope'] = hor_slope
        df_frame_mini['ver_slope'] = ver_slope
        
        hor_lines, ver_lines = frame.group_lines()
        frame.draw_lines_on_image(image, width, height, filename=frame_num+'_grouped.jpeg')
        vi, hi = frame.find_index()
        print('Finished indexing calculations for frame', frame_num)
        if len(vi) != 0:
            df_frame_mini['yi_max'] = np.max(vi)
            df_frame_mini['yi_min'] = np.min(vi)
        if len(hi) != 0:
            df_frame_mini['xi_max'] = np.max(vi)
            df_frame_mini['xi_min'] = np.min(vi)
            
        # generate maps
        frame.generate_map_xy()
        maps_xy.append(frame.map_xy)
        frame.generate_map_dxdy(cf.params['dxdy_spacing'])  #4x spacing
        df_frame_mini['dxdy_spacing'] = cf.params['dxdy_spacing']
        maps_dxdy.append(frame.map_dxdy)
        #frames.append(frame)
        
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
        
        # Write results to dataframe and csv
        mini_df = pd.DataFrame({'x' : x, 'y' : y, 'size' : size, 'xi' : xi, 'yi' : yi})
        mini_df['frame_num'] = frame_num
        cnt += 1
        df = pd.concat([df, mini_df])
        df_frame = pd.concat([df_frame, df_frame_mini])
        print('Total frames processed is', str(cnt), '/', len(image_files))
        
    #determine center dot and FOV, mark outliers
    median_center_x = np.median(df_frame['center_dot_x'])  #median of center locations, use this to determine center dot outlier
    median_center_y = np.median(df_frame['center_dot_y'])
    if (median_center_x > 2/3 * width) or (median_center_x < 1/3 * width) or (median_center_y > 2/3 * height) or (median_center_y < 1/3 * height): 
        raise Exception("Error: Center dot outside of ROI (middle 1/3)")
    df_frame['d_center_dot'] = np.nan
    df_frame['d_fov_center'] = np.nan
    df_frame['flag_center_dot_outlier'] = 0
    df_frame['flag_fov_dot_outlier'] = 0
    for frame in range(len(df_frame)):
        # determine if center dot is outlier based on distance to the median location, if > 50px, mark as outlier
        df_frame.loc[frame, 'd_center_dot'] = np.sqrt((df_frame['center_dot_x'].iloc[frame] - median_center_x)**2 + (df_frame['center_dot_y'].iloc[frame] - median_center_y)**2)
        if df_frame['d_center_dot'].iloc[frame] > 50:
            df_frame['flag_center_dot_outlier'].iloc[frame] = 1
            print ('Warning: center dot outlier detected on frame# ', str(df_frame['frame_num'].iloc[frame]))

        # determine if FOV dot is outlier, if d < 25px, mark as outlier, if y distance > 200px, outlier   
        df_frame.loc[frame, 'd_fov_center'] = np.sqrt((df_frame['fov_dot_x'].iloc[frame] - df_frame['center_dot_x'].iloc[frame])**2 + (df_frame['fov_dot_y'].iloc[frame] - df_frame['center_dot_y'].iloc[frame])**2)        
        if (df_frame['d_fov_center'].iloc[frame] < 25) or (np.abs(df_frame['fov_dot_y'].iloc[frame] - df_frame['center_dot_y'].iloc[frame])) > 200:
            df_frame.loc[frame, 'flag_fov_dot_outlier'] = 1
            print ('Warning: FOV dot outlier detected on frame# ', str(df_frame['frame_num'].iloc[frame]))
  
    df.to_csv(csv_file)
    df_frame.to_csv(csv_file_frame)
    
    # Map Generation
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2)
    
    #find middle frame by min(d_fov_center)
    df_frame = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0)] # filter out outlier frames
    min_d_fov_center = np.min(df_frame['d_fov_center'])
    middle_frame_index = int(df_frame['index'][df_frame['d_fov_center'] == min_d_fov_center].tolist()[0])
    summary= df_frame[df_frame['index'] == middle_frame_index].to_dict(orient='records')[0]  # generate summary dict starting w middle frame info 
    
    map_dxdy_median = calc_median_map(maps_dxdy, min_num=1)
    unitspacing_xy = map_dxdy_median[60, 60] / cf.params['dxdy_spacing']
    
    [xi_fov, yi_fov] = [summary['fov_dot_x'] - summary['center_dot_x'], summary['fov_dot_y'] - summary['center_dot_y']] / unitspacing_xy
    summary['xi_fov'] = xi_fov
    summary['yi_fov'] = yi_fov
    print ('Middle frame found: #', str(summary['frame_num']),'; FOV dot @index ', [xi_fov, yi_fov])
    
    #Normalize map_dxdy for local PS 
    map_dxdy_norm = maps_dxdy[middle_frame_index] / map_dxdy_median
    map_dxdy_norm_fov = offset_map_fov(map_dxdy_norm, xi_fov, yi_fov)
    summary_local = calc_parametrics_local(map_dxdy_norm_fov, map_fov)
    summary.update(summary_local)
    try: 
        plot_map_norm(map_dxdy_norm_fov)
    except ValueError:
        print('Error: plot local PS map')
    
    #Global PS
    map_distance = calc_distance(maps_xy[middle_frame_index], maps_xy[middle_frame_index][60, 60, :])
    map_unit = map_distance / map_fov
   
    for i in [0, -1]: #first and last frame
        if i == 0:
            label = 'Right Gaze'
        else:
            label = 'Left Gaze'
        #frame_num = frame_nums[i]
        frame_num = df_frame['frame_num'].values[i]
        index = int(df_frame['index'].values[i])
        map_delta_global = maps_xy[index] - maps_xy[middle_frame_index]
        map_distance_global = calc_distance(map_delta_global, map_delta_global[60, 60, :])
        map_global = map_distance_global  / map_unit
        map_global = offset_map_fov(map_global, xi_fov, yi_fov)
        summary_global = calc_parametrics_global (map_global, map_fov, label, frame_num)
        summary.update(summary_global)
        try:
            plot_map_global(map_global, fname_str=label)
        except ValueError:
            print('Error: plot global PS map') 

    with open(csv_file_summary, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, summary.keys())
        w.writeheader()
        w.writerow(summary)
    
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
   
