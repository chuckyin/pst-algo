#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import os
import sys
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config.config as cf

from config.logging import logger
from numpy.lib.stride_tricks import sliding_window_view


sys.dont_write_bytecode = True # Disables __pycache__
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='All-NaN axis encountered')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')


def calc_median_map(maps, min_num=3):
    #calculate the median map of a maps (list), with min number of valid elements (will output nan if < min_num)
    cnt = 0
    maps_arr = np.asarray(maps)
    l, xi, yi, dim = np.shape(maps_arr)
    output = np.empty((xi, yi, dim))
    if min_num > l:
        min_num = l
        logger.warning('Warning: min number of valid elements reduce to: %d', l)
    
    for i in range(xi):
        for j in range(yi):
            for k in range(dim):
                if np.count_nonzero(~np.isnan(maps_arr[:, i, j, k])) >= min_num:
                    output[i, j, k] = np.nanmedian(maps_arr[:, i, j, k])
                    cnt += 1
                else:
                    output[i,j,k] = np.nan
    logger.info('Median map calculation complete with %d dots' , cnt)                
    return output


def plot_map_norm(map_norm, levels=None, cmap='coolwarm', fname_str=''):
    xi_full, yi_full, _ = map_norm.shape
    xi_range = int((xi_full - 1) / 2)
    yi_range = int((yi_full - 1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full), np.linspace(-yi_range, yi_range, yi_full))
    
    for i in [0, 1]: # x, y
        if i == 0:
            axis = 'X'
        else:
            axis = 'Y'
        fig = plt.figure(figsize=(5, 5), dpi=200) 
        ax = fig.add_subplot(111)
        plt.title(fname_str + axis)
        plt.xlabel('xi')
        plt.ylabel('yi')
        plt.contourf(X, Y, map_norm[:, :, i].T, levels=levels, cmap=cmap, extend='both')  #cmap ='coolwarm','bwr'
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
        plt.savefig(os.path.join(cf.output_path, fname_str + axis + '.png'))

    
def plot_map_global(map_global):   
    xi_full, yi_full = map_global.shape
    xi_range = int((xi_full - 1) / 2)
    yi_range = int((yi_full - 1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full), np.linspace(-yi_range, yi_range, yi_full))

    fig = plt.figure(figsize=(5, 5), dpi=200) 
    ax = fig.add_subplot(111)
    plt.title('Global PS Map')
    plt.xlabel('xi')
    plt.ylabel('yi')
    levels = np.linspace(-1.1, 1.1, 12)
    plt.contourf(X, Y, map_global[:, :].T, levels=levels, cmap='coolwarm', extend ='max')  #cmap ='coolwarm','bwr'
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
    plt.savefig(os.path.join(cf.output_path, 'Global_PS_Map.png'))       

    
def offset_map_fov (map_input, xi_fov, yi_fov, x_shift=0, y_shift=0):
    def shift_fillnan (arr, shift, axis):
        arr = np.roll(arr, shift, axis)
        if axis == 0:
            if shift >= 0:
                arr[:shift, :] = np.nan
            else:
                arr[shift:, :] = np.nan
        elif axis == 1:
            if shift >= 0:
                arr[:, :shift] = np.nan
            else:    
                arr[:, shift:] = np.nan
        
        return arr    
    
    #offset map and create FOV map (center of display)
    xi_fov = xi_fov.astype('float')
    yi_fov = yi_fov.astype('float')

    map_output = shift_fillnan(map_input, int(np.round(-xi_fov + x_shift)), axis=0) # shift map relative to FOV 
    map_output = shift_fillnan(map_output, int(np.round(-yi_fov + y_shift)), axis=1)
    
    return map_output

    
def calc_local_ps_kpi(map_local, map_type=''):
    summary = {}
    #calculate the parametrics based on the map
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning

    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    axis = ['X', 'Y']
    for i in range(len(radii)):
        zone = (map_fov > start_r) * (map_fov <= radii[i])       
        for j in range(len(axis)):            
            mapp = map_local[:, :, j].T

            summary[map_type + '_rms_d' + axis[j] + '_' + str(radii[i])] = np.nanstd(mapp[zone]) * 100
            summary[map_type + '_pct99_d' + axis[j] + '_' + str(radii[i])] = (np.nanpercentile(mapp[zone], 99) - 1) * 100
            summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i])] = (np.nanpercentile(mapp[zone], 1) - 1) * 100
            summary[map_type + '_pp_d' + axis[j] + '_' + str(radii[i])] = summary[map_type + '_pct99_d' + axis[j]
                                        + '_' + str(radii[i])] - summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i])]
            if radii[i] in [35, 45]:
                # Split map further to compare Nasal and Temporal Zones
                xx_L, yy_L = np.meshgrid(np.linspace(-60, 0, 61), np.linspace(-60, 60, 121))
                xx_R, yy_R = np.meshgrid(np.linspace(0, 60, 61), np.linspace(-60, 60, 121))
                map_fov_L = np.sqrt(xx_L ** 2 + yy_L ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
                map_fov_R = np.sqrt(xx_R ** 2 + yy_R ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
                zone_L = np.concatenate(((map_fov_L > start_r) * (map_fov_L <= radii[i]), np.zeros((121, 60), dtype=bool)), axis=1)
                zone_R = np.concatenate((np.zeros((121, 60), dtype=bool), (map_fov_R > start_r) * (map_fov_R <= radii[i])), axis=1)
                
                summary[map_type + '_pct99_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(mapp[zone_L], 99) - 1) * 100
                summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(mapp[zone_L], 1) - 1) * 100
                summary[map_type + '_rms_d' + axis[j] + '_' + str(radii[i]) + '_L'] = np.nanstd(mapp[zone_L]) * 100
                summary[map_type + '_pp_d' + axis[j] + '_' + str(radii[i]) + '_L'] = summary[map_type + '_pct99_d' + axis[j]
                                            + '_' + str(radii[i]) + '_L'] - summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L']

                summary[map_type + '_pct99_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 99) - 1) * 100
                summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 1) - 1) * 100
                summary[map_type + '_rms_d' + axis[j] + '_' + str(radii[i]) + '_R'] = np.nanstd(mapp[zone_R]) * 100
                summary[map_type + '_pp_d' + axis[j] + '_' + str(radii[i]) + '_R'] = summary[map_type + '_pct99_d' + axis[j]
                                            + '_' + str(radii[i]) + '_R'] - summary[map_type + '_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R']                
        start_r = radii[i]
    
    return summary


def calc_global_ps_kpi(map_global):
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning

    #calculate the parametrics based on the map    
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    
    #define zones
    for i in range(len(radii)):
        zone =  (map_fov > start_r) * (map_fov <= radii[i])
        start_r = radii[i]
        summary['global_' + 'max_' + str(radii[i])] = np.nanmax(map_global[zone])
        summary['global_' + 'pct99_' + str(radii[i])] = np.nanpercentile(map_global[zone], 99)
        summary['global_' + 'pct1_' + str(radii[i])] = np.nanpercentile(map_global[zone], 1)
        summary['global_' + 'pp_' + str(radii[i])] = summary['global_' + 'pct99_' + str(radii[i])] - summary['global_' + 'pct1_' + str(radii[i])]
        summary['global_' + 'median_' + str(radii[i])] = np.nanpercentile(map_global[zone], 50)

    return summary


def normalized_map(maps_dxdy, middle_frame_index, xi_fov, yi_fov, map_dxdy_median, params):
    summary = {}
    middle_map_dxdy_norm = maps_dxdy[middle_frame_index] / map_dxdy_median # Normalize map_dxdy
    middle_map_dxdy_norm_fov = offset_map_fov(middle_map_dxdy_norm, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
    if params['filter_percent'] > 0:
        # Filter resultant map by removing filter_percent
        axis = ['X', 'Y']
        filtered_middle_map = np.empty(np.shape(middle_map_dxdy_norm_fov))
        filtered_middle_map.fill(np.nan)
        for j in range(len(axis)):
            map_mid = middle_map_dxdy_norm_fov[:, :, j]
            upper = 1 + (params['filter_percent'] / 100)
            lower = 1 - (params['filter_percent'] / 100)
            res_mid = np.where(((map_mid > upper) | (map_mid < lower)) & (upper > lower), np.nan, map_mid)
            filtered_middle_map[:, :, j] = res_mid
        plot_map_norm(filtered_middle_map, levels=np.linspace(0.95, 1.05, 11), cmap='seismic', fname_str='Filtered_Normalized_Map_d') 
        summary_norm = calc_local_ps_kpi(filtered_middle_map, map_type='norm_map')
        summary_pp = pp_kernel_map(filtered_middle_map, params, map_type='Normalized Map')
    else:
        plot_map_norm(middle_map_dxdy_norm_fov, levels=np.linspace(0.95, 1.05, 11), cmap='seismic', fname_str='Normalized_Map_d')  
        summary_norm = calc_local_ps_kpi(middle_map_dxdy_norm_fov, map_type='norm_map')
        summary_pp = pp_kernel_map(middle_map_dxdy_norm_fov, params, map_type='Normalized Map')

    summary.update(**summary_norm, **summary_pp)
    
    return summary     


def pp_kernel_map(map_norm, params, map_type=''):
    k = params['kernel_pp_size'] # window size for peak-to-peak calculations
    axis = ['X', 'Y']
    pp = np.empty(np.shape(map_norm))
    pp.fill(np.nan)
    for j in range(len(axis)):
        min_map = np.nanmin(sliding_window_view(map_norm[:, :, j], window_shape=(k, k)), axis=(2, 3))
        max_map = np.nanmax(sliding_window_view(map_norm[:, :, j], window_shape=(k, k)), axis=(2, 3))
        c_map = sliding_window_view(map_norm[:, :, j], window_shape=(k, k))[:, :, int(k / 2), int(k / 2)]

        pp_map = np.nanmax([max_map - c_map, c_map - min_map], axis=0)
        pp[:, :, j] = np.pad(pp_map, pad_width=int(k / 2), mode='constant', constant_values=np.nan)   

    plot_map_norm(pp, levels=np.linspace(0, 0.1, 11), cmap='coolwarm', fname_str='Peak-to-Peak '+map_type)

    # Calculate KPIs
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    axis = ['X', 'Y']
    for i in range(len(radii)):
        zone = (map_fov > start_r) * (map_fov <= radii[i])       
        for j in range(len(axis)):            
            mapp = pp[:, :, j].T

            summary['kernel_map_rms_d' + axis[j] + '_' + str(radii[i])] = np.nanstd(mapp[zone]) * 100
            summary['kernel_map_pct99_d' + axis[j] + '_' + str(radii[i])] = np.nanpercentile(mapp[zone], 99) * 100
            summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i])] = np.nanpercentile(mapp[zone], 1) * 100
            summary['kernel_map_pp_d' + axis[j] + '_' + str(radii[i])] = summary['kernel_map_pct99_d' + axis[j]
                                        + '_' + str(radii[i])] - summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i])]
            if radii[i] in [35, 45]:
                # Split map further to compare Nasal and Temporal Zones
                xx_L, yy_L = np.meshgrid(np.linspace(-60, 0, 61), np.linspace(-60, 60, 121))
                xx_R, yy_R = np.meshgrid(np.linspace(0, 60, 61), np.linspace(-60, 60, 121))
                map_fov_L = np.sqrt(xx_L ** 2 + yy_L ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
                map_fov_R = np.sqrt(xx_R ** 2 + yy_R ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
                zone_L = np.concatenate(((map_fov_L > start_r) * (map_fov_L <= radii[i]), np.zeros((121, 60), dtype=bool)), axis=1)
                zone_R = np.concatenate((np.zeros((121, 60), dtype=bool), (map_fov_R > start_r) * (map_fov_R <= radii[i])), axis=1)
                
                summary['kernel_map_pct99_d' + axis[j] + '_' + str(radii[i]) + '_L'] = np.nanpercentile(mapp[zone_L], 99) * 100
                summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L'] = np.nanpercentile(mapp[zone_L], 1) * 100
                summary['kernel_map_rms_d' + axis[j] + '_' + str(radii[i]) + '_L'] = np.nanstd(mapp[zone_L]) * 100
                summary['kernel_map_pp_d' + axis[j] + '_' + str(radii[i]) + '_L'] = summary['kernel_map_pct99_d' + axis[j]
                                            + '_' + str(radii[i]) + '_L'] - summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L']

                summary['kernel_map_pct99_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 99) - 1) * 100
                summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 1) - 1) * 100
                summary['kernel_map_rms_d' + axis[j] + '_' + str(radii[i]) + '_R'] = np.nanstd(mapp[zone_R]) * 100
                summary['kernel_map_pp_d' + axis[j] + '_' + str(radii[i]) + '_R'] = summary['kernel_map_pct99_d' + axis[j]
                                            + '_' + str(radii[i]) + '_R'] - summary['kernel_map_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R']                
        start_r = radii[i]

    return summary


def average_map(maps_dxdy, df_frame, map_dxdy_median, params):
    maps_dxdy_norm = maps_dxdy / map_dxdy_median # Normalize avg_map_dxdy     
    maps_dxdy_fov_norm = [offset_map_fov(maps_dxdy_norm[frame_idx], df_frame.loc[frame_idx, 'xi_fov'], df_frame.loc[frame_idx, 'yi_fov'], params['map_x_shift'], params['map_y_shift']) \
                        for frame_idx in range(len(df_frame)) if (df_frame.loc[frame_idx, 'flag_center_dot_outlier'] == 0) and 
                        (df_frame.loc[frame_idx, 'flag_fov_dot_outlier'] == 0) and (df_frame.loc[frame_idx, 'flag_slope_outlier'] == 0)]
    
    print('Number of frames used for averaging:', len(maps_dxdy_fov_norm))

    if params['filter_percent'] > 0 and params['enable_all_saving']:
        # Filter resultant map by removing filter_percent
        axis = ['X', 'Y']
        filtered_maps = np.empty(np.shape(maps_dxdy_fov_norm))
        filtered_maps.fill(np.nan)
        for frame in range(len(maps_dxdy_fov_norm)):
            for j in range(len(axis)):
                mapp = maps_dxdy_fov_norm[frame][:, :, j]
                upper = 1 + (params['filter_percent'] / 100)
                lower = 1 - (params['filter_percent'] / 100)
                res_map = np.where(((mapp > upper) | (mapp < lower)) & (upper > lower), np.nan, mapp)
                filtered_maps[frame, :, :, j] = res_map
            plot_map_norm(filtered_maps[frame], levels=np.linspace(0.985, 1.015, 7), fname_str='Filtered_'+str(frame)+'_Map_d')
        with warnings.catch_warnings():
            avg_map_dxdy_fov_norm = np.nanmean(filtered_maps, axis=0)
    elif params['enable_all_saving']:
        for frame in range(len(maps_dxdy_fov_norm)): 
            plot_map_norm(maps_dxdy_fov_norm[frame], levels=np.linspace(0.985, 1.015, 7), fname_str=str(frame)+'_Map_d')
        avg_map_dxdy_fov_norm = np.nanmean(maps_dxdy_fov_norm, axis=0)
    elif params['filter_percent'] > 0:
        # Filter resultant map by removing filter_percent
        axis = ['X', 'Y']
        filtered_maps = np.empty(np.shape(maps_dxdy_fov_norm))
        filtered_maps.fill(np.nan)
        for frame in range(len(maps_dxdy_fov_norm)):
            for j in range(len(axis)):
                mapp = maps_dxdy_fov_norm[frame][:, :, j]
                upper = 1 + (params['filter_percent'] / 100)
                lower = 1 - (params['filter_percent'] / 100)
                res_map = np.where(((mapp > upper) | (mapp < lower)) & (upper > lower), np.nan, mapp)
                filtered_maps[frame, :, :, j] = res_map
        avg_map_dxdy_fov_norm = np.nanmean(filtered_maps, axis=0)
      
    plot_map_norm(avg_map_dxdy_fov_norm, levels=np.linspace(0.985, 1.015, 7), fname_str='Average_Map_d')
    summary = calc_local_ps_kpi(avg_map_dxdy_fov_norm, map_type='avg_map')      

    return summary 


def eval_KPIs(dots, frames, params, summary_old, middle_frame_index, maps_dxdy):
    def detect_center_shift(dots, params):
        try:
            frame1000 = cv2.imread(os.path.join(cf.input_path, '1000.tiff'))
        except FileNotFoundError:
            logger.error('1000.tiff was not found. Must be using older configuration without frames 1000/1001')
            return -1

        center1001x = dots.loc[(dots['frame_num'] == '1001') & (dots['xi'] == 0) & (dots['yi'] == 0)]['x'].astype('int')
        center1001y = dots.loc[(dots['frame_num'] == '1001') & (dots['xi'] == 0) & (dots['yi'] == 0)]['y'].astype('int')
        center1001r = dots.loc[(dots['frame_num'] == '1001') & (dots['xi'] == 0) & (dots['yi'] == 0)]['size'] / 2

        frame1000_gray = cv2.cvtColor(frame1000, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(frame1000_gray)
        cv2.circle(mask, (int(center1001x), int(center1001y)), int(center1001r), 255, -1)
        dotmask = cv2.bitwise_or(frame1000_gray, mask)

        reg = dotmask[int(center1001y - 2 * center1001r) : int(center1001y + 2 * center1001r), int(center1001x - 2 * center1001r) : int(center1001x + 2 * center1001r)]
        _, thresh = cv2.threshold(reg, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if params['enable_all_saving']:
            cv2.drawContours(reg, contours, -1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(cf.output_path, 'overlapped_center_dot.jpeg'), reg)

        if len(contours) > 1:
            area = max([cv2.contourArea(contour) for contour in contours])
        else:
            area = cv2.contourArea(contours[0])
        circ = np.pi * float(center1001r * center1001r)

        return (area / circ)

    
    map_dxdy_median = calc_median_map(maps_dxdy, min_num=1)
    assert(params['dxdy_spacing'] > 0)
    unitspacing_xy = map_dxdy_median[60, 60] / params['dxdy_spacing']
    
    frames['xi_fov'] = ((frames['fov_dot_x'] - frames['center_dot_x']) / unitspacing_xy[0]).astype('float64')
    frames['yi_fov'] = ((frames['fov_dot_y'] - frames['center_dot_y']) / unitspacing_xy[1]).astype('float64')

    summary_old['algo_version'] = cf.get_version()
    summary_old['xi_fov'] = (summary_old['fov_dot_x'] - summary_old['center_dot_x']) / unitspacing_xy[0]
    summary_old['yi_fov'] = (summary_old['fov_dot_y'] - summary_old['center_dot_y']) / unitspacing_xy[1]
    logger.info('Generated Maps are shifted by (%0.2f, %0.2f)', summary_old['map_x_shift'], summary_old['map_y_shift'])
    logger.info('FOV dot is indexed at (%0.2f, %0.2f)', summary_old['xi_fov'], summary_old['yi_fov'])

    if params['detect_center_shift']:
        # Detect whether there is an unacceptable center dot shift between frames 1000 and 1001
        summary_old['ratio'] = detect_center_shift(dots, params)
        logger.info('Detected center dot shift confidence was %0.2f', summary_old['ratio'])
    
    # Local PS
    try:
        summary_norm = normalized_map(maps_dxdy, middle_frame_index, summary_old['xi_fov'], summary_old['yi_fov'], map_dxdy_median, params)
        summary_avg = average_map(maps_dxdy, frames, map_dxdy_median, params)
        summary_local = pd.concat([summary_old, pd.DataFrame(summary_norm.values()).T, pd.DataFrame(summary_avg.values()).T], axis=1, ignore_index=True)
        summary_local.columns = summary_old.columns.tolist() + [col for col in summary_norm.keys()] + [col for col in summary_avg.keys()]
    except ValueError:
        logger.error('Error calculating and plotting local PS map')
        
    # Global PS
    no_center_frame = frames.drop(frames.loc[frames['frame_num'] == '1001'].index)
    no_center_frame.drop(no_center_frame.loc[no_center_frame['num_total_outlier'] > 0].index, inplace=True)

    try:
        first_frame_num = no_center_frame.loc[0, 'frame_num']
        last_frame_num = no_center_frame.loc[no_center_frame.index[-1], 'frame_num']

        x = no_center_frame['frame_num'].values.astype('int')
        y = no_center_frame['xi_fov'].values

        coeff = np.polyfit(x, -y, 1)
        logger.info('Linear slope fit: %0.2f, intercept: %0.2f', coeff[0], coeff[1])
        dots['xi_proj'] = dots['xi'] + (coeff[1] + coeff[0] * dots['frame_num'].astype('int'))

        # Find the synthetic frame by slicing and stitching frames
        if coeff[0] > 0:
            # Right to left
            dots.loc[(dots['xi_proj'] >= - params['stitch_size']) & (dots['xi_proj'] <= params['stitch_size']) | 
                ((dots['frame_num'] == last_frame_num) & (dots['xi_proj'] <= params['stitch_size'])) |
                (dots['xi_proj'] >= - params['stitch_size']) & (dots['frame_num'] == first_frame_num), 'xi_mask'] = 1
            dots['xi_mask'].fillna(0, inplace=True)
        else:
            # Left to Right
            dots.loc[(dots['xi_proj'] >= - params['stitch_size']) & (dots['xi_proj'] <= params['stitch_size']) | 
                ((dots['frame_num'] == last_frame_num) & (dots['xi_proj'] >= - params['stitch_size'])) |
                (dots['xi_proj'] <= params['stitch_size']) & (dots['frame_num'] == first_frame_num), 'xi_mask'] = 1
            dots['xi_mask'].fillna(0, inplace=True)

    except KeyError:   # Use default coefficients 
        no_outlier_frame = frames.drop(frames.loc[(frames['frame_num'] == '1001') | (frames['flag_slope_outlier'] == 1)
                     | (frames['flag_fov_dot_outlier'] == 1) | (frames['flag_center_dot_outlier'] == 1)].index)
        no_outlier_frame.reset_index(inplace=True)
        first_frame_num = no_outlier_frame.loc[0, 'frame_num']
        last_frame_num = no_outlier_frame.loc[no_outlier_frame.index[-1], 'frame_num']
        y_first = no_outlier_frame.loc[0, 'xi_fov']
        y_last = no_outlier_frame.loc[no_outlier_frame.index[-1], 'xi_fov']
        if (y_first - y_last) < 0:
            # Right to Left
            dots['xi_proj'] = dots['xi'] + (-23.5 + 0.26 * dots['frame_num'].astype('int'))
            dots.loc[(dots['xi_proj'] >= - params['stitch_size']) & (dots['xi_proj'] <= params['stitch_size']) | 
                ((dots['frame_num'] == last_frame_num) & (dots['xi_proj'] <= params['stitch_size'])) |
                (dots['xi_proj'] >= - params['stitch_size']) & (dots['frame_num'] == first_frame_num), 'xi_mask'] = 1
            dots['xi_mask'].fillna(0, inplace=True)
        else:
            # Left to Right
            dots['xi_proj'] = dots['xi'] + (20.2 - 0.255 * dots['frame_num'].astype('int'))
            dots.loc[(dots['xi_proj'] >= - params['stitch_size']) & (dots['xi_proj'] <= params['stitch_size']) | 
                ((dots['frame_num'] == last_frame_num) & (dots['xi_proj'] >= - params['stitch_size'])) |
                (dots['xi_proj'] <= params['stitch_size']) & (dots['frame_num'] == first_frame_num), 'xi_mask'] = 1
            dots['xi_mask'].fillna(0, inplace=True)

    merged = dots.merge(frames, on=['frame_num'])

    merged['xx'] = merged['x'] - merged['center_dot_x'].astype('float64')
    merged['yy'] = merged['y'] - merged['center_dot_y'].astype('float64')

    merged.set_index(['xi', 'yi'], inplace=True)
    merged['xx0'] = merged.loc[merged['xi_mask'] == 1].groupby(['xi', 'yi'])['xx'].median()
    merged['yy0'] = merged.loc[merged['xi_mask'] == 1].groupby(['xi', 'yi'])['yy'].median()
    merged = merged.reset_index()

    merged['dxx'] = merged['xx0'] - merged['xx']
    merged['dyy'] = merged['yy0'] - merged['yy']
    merged['drr'] = np.sqrt(merged['dxx'] ** 2 + merged['dyy'] ** 2)
    merged['rr_std'] = np.sqrt(merged['xx'] ** 2 + merged['yy'] ** 2) / np.sqrt(merged['xi'] ** 2 + merged['yi'] ** 2)
    merged['drr_norm'] = merged['drr'] / merged['rr_std']
    merged['rr'] = np.sqrt(merged['xx'] ** 2 + merged['yy'] ** 2)
    merged['rr0'] = np.sqrt(merged['xx0'] ** 2 + merged['yy0'] ** 2)
    merged['rr0_unit'] = merged['rr0'] / np.sqrt(merged['xi'] ** 2 + merged['yi'] ** 2)
    merged['drr_radial'] = (merged['rr0'] - merged['rr']) / merged['rr0_unit']

    # Generate map
    map_drr = np.empty((60 * 2 + 1, 60 * 2 + 1)) * np.nan
    ref = merged.loc[merged['frame_num'] == '1001']
    xi_fov = frames.loc[frames['frame_num'] == '1001']['xi_fov']
    yi_fov = frames.loc[frames['frame_num'] == '1001']['yi_fov']

    #----------REWRITE-------------------------------------#
    # xil = (ref['xi'].astype('Int64').values + 60).tolist()
    # yil = (ref['yi'].astype('Int64').values + 60).tolist()
    # for i, (xi, yi) in enumerate(zip(xil, yil)):
    #     if (not pd.isna(xi)) and (not pd.isna(yi)):
    #         map_drr[xi, yi] = ref['drr_radial'].iloc[i]
    #------------------------------------------------------#

    for i, (xi, yi) in enumerate(zip(ref['xi'].values, ref['yi'].values)):
        if ~np.isnan(xi) and ~np.isnan(yi) and (xi<=61) and (yi <= 61):
            map_drr[int(xi) + 60, int(yi) + 60] = ref['drr_radial'].iloc[i]

    # Save map values as csv
    np.savetxt(os.path.join(cf.output_path, 'map_drr.csv'), map_drr.T, delimiter=',')

    map_global = offset_map_fov(map_drr, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
    plot_map_global(map_global)
    summary_global = calc_global_ps_kpi(map_global)

    cols = [col for col in summary_global.keys()]
    summary_global = pd.DataFrame(summary_global.values()).T
    summary_global.columns = cols
    summary_df = pd.concat([summary_local, summary_global], axis=1, ignore_index=True)
    summary_df.columns = summary_local.columns.tolist() + summary_global.columns.tolist()
        
    return summary_df
    #return summary_local

def find_middle_frame(df_frame):   
    # find middle frame by min(d_fov_center)
    try:
        df_frame_no_outliers = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0) & (df_frame['flag_slope_outlier'] == 0)] # filter out outlier frames
        min_d_fov_center = np.min(df_frame_no_outliers['dist_fov_center'])
        middle_frame_index = df_frame.loc[df_frame['dist_fov_center'] == min_d_fov_center].index[0]
    except:
        logger.exception('Error: Cannot find a suitable middle frame -- too many outliers -- Will use frame 6 for calculations')
        middle_frame_index = 6
    
    return middle_frame_index
    

def find_outliers(df_frame, params, width, height):
    # determine center dot and FOV outliers
    median_center_x = np.median(df_frame['center_dot_x'])  # median of center dot locations, use this to determine center dot outlier
    median_center_y = np.median(df_frame['center_dot_y'])
    if (median_center_x > 2/3 * width) or (median_center_x < 1/3 * width) or (median_center_y > 2/3 * height) or (median_center_y < 1/3 * height): 
        logger.exception('Error: Center dot outside of ROI (middle 1/3)')
    df_frame['dist_center_dot'] = np.nan
    df_frame['dist_fov_center'] = np.nan
    df_frame['flag_center_dot_outlier'] = 0
    df_frame['flag_fov_dot_outlier'] = 0
    df_frame['flag_slope_outlier'] = 0
    num_outliers = 0 
    num_fov_outliers = 0
    num_slope_outliers = 0
    for i in range(len(df_frame.index)):
        # determine if center dot is outlier based on distance to the median location, if > 50px, mark as outlier
        df_frame.loc[i, 'dist_center_dot'] = np.sqrt((df_frame.loc[i, 'center_dot_x'] - median_center_x) ** 2 + (df_frame.loc[i, 'center_dot_y'] - median_center_y) ** 2)
        if df_frame.loc[i, 'dist_center_dot'] > params['median_center_dot_dist']:
            df_frame.loc[i, 'flag_center_dot_outlier'] = 1
            num_outliers += 1
            logger.warning('Center dot outlier detected on frame #%s', df_frame.loc[i, 'frame_num'])

        # determine if FOV dot is outlier, if d < 25px, mark as outlier, if y distance > 200px, outlier   
        df_frame.loc[i, 'dist_fov_center'] = np.sqrt((df_frame.loc[i,'fov_dot_x'] - df_frame.loc[i, 'center_dot_x']) ** 2 \
                                                    + (df_frame.loc[i,'fov_dot_y'] - df_frame.loc[i, 'center_dot_y']) ** 2)        
        if (df_frame.loc[i, 'dist_fov_center'] < params['delta_dist_fov_center']) or \
                    ((np.abs(df_frame.loc[i, 'fov_dot_y'] - df_frame.loc[i, 'center_dot_y'])) > params['delta_y_fov_center']) or \
                    ((df_frame.loc[i, 'fov_dot_x'] == width / 2) and (df_frame.loc[i, 'fov_dot_y'] == height / 2)):
            df_frame.loc[i, 'flag_fov_dot_outlier'] = 1
            num_fov_outliers += 1
            logger.warning('FOV dot outlier detected on frame #%s', df_frame.loc[i, 'frame_num'])
        
        # check slope
        if (df_frame.loc[i, 'hor_slope'] != None) and (df_frame.loc[i, 'ver_slope'] != None):    
            if (np.abs(df_frame.loc[i, 'hor_slope']) > 0.1) or (np.abs(df_frame.loc[i, 'ver_slope']) > 0.1):
                df_frame.loc[i, 'flag_slope_outlier'] = 1
                num_slope_outliers += 1
                logger.warning('Slope outlier detected on frame #%s', df_frame.loc[i, 'frame_num'])

    df_frame['num_frames_processed'] = len(df_frame.index)
    df_frame['num_center_dot_outlier'] = num_outliers
    df_frame['num_fov_dot_outlier'] = num_fov_outliers
    df_frame['num_slope_outlier'] = num_slope_outliers
    df_frame['num_total_outlier'] = num_outliers + num_fov_outliers + num_slope_outliers
    