#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config.config as cf

from config.logging import logger
from numpy.lib.stride_tricks import sliding_window_view
from multipledispatch import dispatch


sys.dont_write_bytecode = True # Disables __pycache__


def calc_median_map(maps, min_num=3):
    #calculate the median map of a maps (list), with min number of valid elements (will output nan if <min_num)
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


def plot_map_norm(map_norm, levels=None, fname_str=''):
    if map_norm.ndim > 2: # N 2D xy maps
        xi_full, yi_full, _ = map_norm.shape
        xi_range = int((xi_full - 1) / 2)
        yi_range = int((yi_full - 1) / 2)
        X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full),np.linspace(-yi_range, yi_range, yi_full))
        
        for i in [0, 1]: # x, y
            if i == 0:
                axis = 'X'
            else:
                axis = 'Y'
            fig = plt.figure(figsize=(5, 5), dpi=200) 
            ax = fig.add_subplot(111)
            plt.title(fname_str + axis)
            #plt.title('Normalized ' + axis + '-Spacing Change')
            plt.xlabel('xi')
            plt.ylabel('yi')
            levels = np.linspace(0.95, 1.05, 11)
            plt.contourf(X, Y, map_norm[:, :, i].T, levels=levels, cmap='seismic', extend='both')  #cmap ='coolwarm','bwr'
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
            plt.savefig(os.path.join(cf.output_path, fname_str + axis +'.png'))
        
    else:
        xi_full, yi_full = map_norm.shape
        xi_range = int((xi_full - 1) / 2)
        yi_range = int((yi_full - 1) / 2)
        X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full),np.linspace(-yi_range, yi_range, yi_full))
        
        fig = plt.figure(figsize=(5, 5), dpi=200) 
        ax = fig.add_subplot(111)
        plt.title(fname_str)
        plt.xlabel('xi')
        plt.ylabel('yi')
        #levels = np.linspace(0, 16, 5)
        plt.contourf(X, Y, map_norm, levels=levels, cmap='seismic', extend='both')  #cmap ='coolwarm','bwr'
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
        plt.savefig(os.path.join(cf.output_path, fname_str + '.png'))

    
def plot_map_global(map_global, fname_str=''):   
    xi_full, yi_full = map_global.shape
    xi_range = int((xi_full -1) / 2)
    yi_range = int((yi_full -1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range, xi_range, xi_full), np.linspace(-yi_range, yi_range, yi_full))
    #map_global = np.ma.array(map_global, mask=np.isnan(map_global))

    fig = plt.figure(figsize=(5, 5), dpi=200) 
    ax = fig.add_subplot(111)
    plt.title('Global PS Map ' + fname_str + ' [degree]')
    plt.xlabel('xi')
    plt.ylabel('yi')
    levels = np.linspace(0, 1.0, 6)
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
    plt.savefig(os.path.join(cf.output_path, 'Global_PS_Map_' + fname_str + '.png'))       

    
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
    map_output = shift_fillnan(map_input, int(np.round(-xi_fov + x_shift)), axis=0) #shift map relative to FOV 
    map_output = shift_fillnan(map_output, int(np.round(-yi_fov + y_shift)), axis=1)
    
    return map_output

    
def calc_local_peripheral(map_local, map_fov, params):
    #calculate the parametrics based on the map
    summary = {}
    start_r = 25
    radii = [35, 45]  #need to start from small to large
    axis = ['X', 'Y']
    th = [0.5, 1, 5]  #in percent
    k = params['kernel_pp_size'] # window size for peak-to-peak calculations
    
    for i in range(len(radii)):
        zone = (map_fov > start_r) * (map_fov <= radii[i])
        mapp_both_x_L = []
        mapp_both_x_R = []
        for j in range(len(axis)):            
            mapp = map_local[:, :, j].T
            # Split map further to compare Nasal and Temporal Zones
            summary['local_areatotal_d' + axis[j] + '_' + str(radii[i])] = np.count_nonzero(~np.isnan(mapp[zone]))
            summary['local_max_d' + axis[j] + '_' + str(radii[i])] = (np.nanmax(mapp[zone]) - 1) * 100
            summary['local_min_d' + axis[j] + '_' + str(radii[i])] = (np.nanmin(mapp[zone]) - 1) * 100
            summary['local_rms_d' + axis[j] + '_' + str(radii[i])] = (np.nanstd(mapp[zone])) * 100
            
            xx_L, yy_L = np.meshgrid(np.linspace(-60, 0, 61), np.linspace(-60, 60, 121))
            xx_R, yy_R = np.meshgrid(np.linspace(0, 60, 61), np.linspace(-60, 60, 121))
            map_fov_L = np.sqrt(xx_L ** 2 + yy_L ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
            map_fov_R = np.sqrt(xx_R ** 2 + yy_R ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
            zone_L = np.concatenate(((map_fov_L > start_r) * (map_fov_L <= radii[i]), np.zeros((121, 60), dtype=bool)), axis=1)
            zone_R = np.concatenate((np.zeros((121, 60), dtype=bool), (map_fov_R > start_r) * (map_fov_R <= radii[i])), axis=1)
            
            summary['local_pct99_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(mapp[zone_L], 99) - 1) * 100
            summary['local_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(mapp[zone_L], 1) - 1) * 100
            summary['local_rms_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanstd(mapp[zone_L])) * 100
            summary['local_pct99_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 99) - 1) * 100
            summary['local_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(mapp[zone_R], 1) - 1) * 100
            summary['local_rms_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanstd(mapp[zone_R])) * 100
            
            summary['local_pp_d' + axis[j] + '_' + str(radii[i]) + '_L'] = summary['local_pct99_d' + axis[j]
                                                    + '_' + str(radii[i]) + '_L'] - summary['local_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L']
            summary['local_pp_d' + axis[j] + '_' + str(radii[i]) + '_R'] = summary['local_pct99_d' + axis[j]
                                                    + '_' + str(radii[i]) + '_R'] - summary['local_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R']                
            # Peak-to-Peak Maps
            zone_mapp_L = np.where(zone_L, mapp, np.nan)
            max_mapp_L = np.max(sliding_window_view(zone_mapp_L, window_shape=(k, k)), axis=(2, 3))
            min_mapp_L = np.min(sliding_window_view(zone_mapp_L, window_shape=(k, k)), axis=(2, 3))
            pp_mapp_L = max_mapp_L - min_mapp_L
            pp_L = np.pad(pp_mapp_L, pad_width=int(k/2), mode='constant', constant_values=np.nan)
            plot_map_norm(pp_L, fname_str='Peak-to-Peak_' + axis[j] + '_map_' + str(radii[i]) + '_degree_zone_L')
            summary['kernel_pct99_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(pp_L, 99) - 1) * 100
            summary['kernel_pct1_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanpercentile(pp_L, 1) - 1) * 100
            summary['kernel_rms_d' + axis[j] + '_' + str(radii[i]) + '_L'] = (np.nanstd(pp_L)) * 100
            
            zone_mapp_R = np.where(zone_R, mapp, np.nan)
            max_mapp_R = np.max(sliding_window_view(zone_mapp_R, window_shape=(k, k)), axis=(2, 3))
            min_mapp_R = np.min(sliding_window_view(zone_mapp_R, window_shape=(k, k)), axis=(2, 3))
            pp_mapp_R = max_mapp_R - min_mapp_R
            pp_R = np.pad(pp_mapp_R, pad_width=int(k/2), mode='constant', constant_values=np.nan)
            plot_map_norm(pp_R, fname_str='Peak-to-Peak_' + axis[j] + '_map_' + str(radii[i]) + '_degree_zone_R')
            summary['kernel_pct99_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(pp_R, 99) - 1) * 100
            summary['kernel_pct1_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanpercentile(pp_R, 1) - 1) * 100
            summary['kernel_rms_d' + axis[j] + '_' + str(radii[i]) + '_R'] = (np.nanstd(pp_R)) * 100
            
            for k in range(len(th)): 
                mapp_pos_L = (mapp >= 1 + th[k] / 100) * zone_L
                mapp_pos_R = (mapp >= 1 + th[k] / 100) * zone_R
                mapp_neg_L = (mapp <= 1 - th[k] / 100) * zone_L
                mapp_neg_R = (mapp <= 1 - th[k] / 100) * zone_R
                mapp_both_L = ((mapp >= 1 + th[k] / 100) + (mapp <= 1 - th[k] / 100)) * zone_L
                mapp_both_R = ((mapp >= 1 + th[k] / 100) + (mapp <= 1 - th[k] / 100)) * zone_R
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pctpos_' + str(radii[i]) + '_L'] = np.count_nonzero(mapp_pos_L)
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pctneg_' + str(radii[i]) + '_L'] = np.count_nonzero(mapp_neg_L)
                summary['local_area_d' + axis[j] + '_th'  + str(th[k]) + 'pct_' + str(radii[i]) + '_L'] = np.count_nonzero(mapp_both_L)
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pctpos_' + str(radii[i]) + '_R'] = np.count_nonzero(mapp_pos_R)
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pctneg_' + str(radii[i]) + '_R'] = np.count_nonzero(mapp_neg_R)
                summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pct_' + str(radii[i]) + '_R'] = np.count_nonzero(mapp_both_R)
                if j == 0: # x maps
                    mapp_both_x_L.append(mapp_both_L)
                    mapp_both_x_R.append(mapp_both_R)
                elif j == 1: # y maps
                    mapp_both_combine_L = mapp_both_L + mapp_both_x_L[k]
                    summary['local_area_combined_th' + str(th[k]) + 'pct_' + str(radii[i]) + '_L'] = np.count_nonzero(mapp_both_combine_L)
                    mapp_both_combine_R = mapp_both_R + mapp_both_x_R[k]
                    summary['local_area_combined_th' + str(th[k]) + 'pct_' + str(radii[i]) + '_R'] = np.count_nonzero(mapp_both_combine_R)
        start_r = radii[i]        
    return summary

def calc_local_central(map_local, map_fov, params):
    #calculate the parametrics based on the map
    summary = {}
    start_r = -1
    radii = 25  
    axis = ['X', 'Y']
    th = [0.5, 1, 5]  #in percent
    k = params['kernel_pp_size'] # window size for peak-to-peak calculations
    
    zone = (map_fov > start_r) * (map_fov <= radii)
    mapp_both_x = []
    for j in range(len(axis)):            
        mapp = map_local[:, :, j].T
        summary['local_areatotal_d' + axis[j] + '_' + str(radii)] = np.count_nonzero(~np.isnan(mapp[zone]))
        summary['local_max_d' + axis[j] + '_' + str(radii)] = (np.nanmax(mapp[zone]) - 1) * 100
        summary['local_min_d' + axis[j] + '_' + str(radii)] = (np.nanmin(mapp[zone]) - 1) * 100
        summary['local_pct99_d' + axis[j] + '_' + str(radii)] = (np.nanpercentile(mapp[zone], 99) - 1) * 100
        summary['local_pct1_d' + axis[j] + '_' + str(radii)] = (np.nanpercentile(mapp[zone], 1) - 1) * 100
        summary['local_rms_d' + axis[j] + '_' + str(radii)] = (np.nanstd(mapp[zone])) * 100
        summary['local_pp_d' + axis[j] + '_' + str(radii)] = summary['local_pct99_d' + axis[j]
                                                + '_' + str(radii)] - summary['local_pct1_d' + axis[j] + '_' + str(radii)]
        # Peak-to-Peak Maps
        zone_mapp = np.where(zone, mapp, np.nan)
        max_mapp = np.max(sliding_window_view(zone_mapp, window_shape=(k, k)), axis=(2, 3))
        min_mapp = np.min(sliding_window_view(zone_mapp, window_shape=(k, k)), axis=(2, 3))
        pp_mapp = max_mapp - min_mapp
        pp = np.pad(pp_mapp, pad_width=int(k/2), mode='constant', constant_values=np.nan)
        plot_map_norm(pp, fname_str='Peak-to-Peak_' + axis[j] + '_map_' + str(radii) + '_degree_zone')
        summary['kernel_rms_d' + axis[j] + '_' + str(radii)] = (np.nanstd(pp)) * 100
        summary['kernel_pct99_d' + axis[j] + '_' + str(radii)] = (np.nanpercentile(pp, 99) - 1) * 100
        summary['kernel_pct1_d' + axis[j] + '_' + str(radii)] = (np.nanpercentile(pp, 1) - 1) * 100
        
        for k in range(len(th)): 
            mapp_pos = (mapp >= 1 + th[k] / 100) * zone
            mapp_neg = (mapp <= 1 - th[k] / 100) * zone
            mapp_both = ((mapp >= 1 + th[k] / 100) + (mapp <= 1 - th[k] / 100)) * zone
            summary['local_area_d' + axis[j] + '_th'+ str(th[k]) + 'pctpos_' + str(radii)] = np.count_nonzero(mapp_pos)
            summary['local_area_d' + axis[j] + '_th'+ str(th[k]) + 'pctneg_' + str(radii)] = np.count_nonzero(mapp_neg)
            summary['local_area_d' + axis[j] + '_th' + str(th[k]) + 'pct_' + str(radii)] = np.count_nonzero(mapp_both)
            if j == 0: # x maps
                mapp_both_x.append(mapp_both)
            elif j == 1: # y maps
                mapp_both_combine = mapp_both + mapp_both_x[k]
                summary['local_area_combined_th' + str(th[k]) + 'pct_' + str(radii)] = np.count_nonzero(mapp_both_combine)

    return summary


def calc_parametrics_global(map_global, map_fov, label, frame_num):
    #calculate the parametrics based on the map    
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    summary['global_' + label + '_frame_num'] = frame_num
    
    #define zones
    for i in range(len(radii)):
        zone =  (map_fov > start_r) * (map_fov <= radii[i])
        start_r = radii[i]
        summary['global_' + label + '_max_' + str(radii[i])] = np.nanmax(map_global[zone])
        summary['global_' + label + '_pct99_' + str(radii[i])] = np.nanpercentile(map_global[zone], 99)
        summary['global_' + label + '_median_' + str(radii[i])] = np.nanpercentile(map_global[zone], 50)

    return summary


@dispatch(pd.DataFrame, dict, int, list, list)
def eval_KPIs(df_frame, params, middle_frame_index, maps_xy, maps_dxdy):
    def calc_distance(map_xy, ref0):
        #calculate distance map, relative to ref0, ref0 could be map of same shape, or x,y coor 1-dim array
        xi_full, yi_full, _ = map_xy.shape

        map_distance = np.empty((xi_full, yi_full))
        map_distance[:,:] = np.nan
        
        map_xy_delta = map_xy - ref0
        
        for i in range(xi_full):
            for j in range(yi_full):
                map_distance[i, j] = np.sqrt((map_xy_delta[i, j, 0] ** 2 + map_xy_delta[i, j, 1] ** 2))
        
        return map_distance

    map_dxdy_median = calc_median_map(maps_dxdy, min_num=1)
    assert(params['dxdy_spacing'] > 0)
    unitspacing_xy = map_dxdy_median[60, 60] / params['dxdy_spacing']
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
    
    summary = df_frame.loc[[middle_frame_index]].to_dict(orient='records')[0]
    (xi_fov, yi_fov) = [summary['fov_dot_x'] - summary['center_dot_x'], summary['fov_dot_y'] - summary['center_dot_y']] / unitspacing_xy
    summary['version'] = cf.get_version()
    summary['xi_fov'] = xi_fov
    summary['yi_fov'] = yi_fov
    logger.info('Middle frame determined to be #%s', summary['frame_num'])
    logger.info('Generated Maps are shifted by (%0.2f, %0.2f)', summary['map_x_shift'], summary['map_y_shift'])
    logger.info('FOV dot is indexed at (%0.2f, %0.2f)', xi_fov, yi_fov)
    
    # Local PS
    try:
        middle_map_dxdy_norm = maps_dxdy[middle_frame_index] / map_dxdy_median # Normalize map_dxdy
        middle_map_dxdy_norm_fov = offset_map_fov(middle_map_dxdy_norm, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
        avg_map_dxdy = np.mean(maps_dxdy, axis=0)
        avg_map_dxdy_fov = offset_map_fov(avg_map_dxdy, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
        if params['filter_percent'] > 0:
            # Filter resultant map by removing filter_percent
            axis = ['X', 'Y']
            filtered_map = np.empty(np.shape(middle_map_dxdy_norm_fov))
            filtered_map.fill(np.nan)
            for j in range(len(axis)):
                mapj = middle_map_dxdy_norm_fov[:, :, j]
                upper = 1 + (params['filter_percent'] / 100)
                lower = 1 - (params['filter_percent'] / 100)
                resj = np.where(((mapj > upper) | (mapj < lower)) & (upper > lower), np.nan, mapj)
                filtered_map[:, :, j] = resj
                # if j == 0:
                #     df = pd.DataFrame(resj)
                #     df.to_csv(os.path.join(cf.output_path,'map_norm_dx_filtered.csv'))
                
            plot_map_norm(filtered_map, fname_str='Normalized_Map_')
            summary_local = calc_local_peripheral(filtered_map, map_fov, params)
            summary.update(summary_local) 
            summary_local = calc_local_central(avg_map_dxdy_fov, map_fov, params)
            summary.update(summary_local)
        else:
            plot_map_norm(middle_map_dxdy_norm_fov, fname_str='Normalized_Map_')
            summary_local = calc_local_peripheral(middle_map_dxdy_norm_fov, map_fov, params)
            summary.update(summary_local)
            summary_local = calc_local_central(avg_map_dxdy_fov, map_fov, params)
            summary.update(summary_local)
    except ValueError:
        logger.error('Error calculating and plotting local PS map')
        pass
    
    # Global PS
    map_distance = calc_distance(maps_xy[middle_frame_index], maps_xy[middle_frame_index][60, 60, :])
    map_unit = map_distance / map_fov + sys.float_info.epsilon # takes care of divide-by-zero warning
    df_frame_no_outliers = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0) & (df_frame['flag_slope_outlier'] == 0)] # filter out outlier frames
      
    for i in [0, -1]: #first and last frame
        if i == 0:
            label = 'Right Gaze'
        else:
            label = 'Left Gaze'
        try:
            idx = df_frame_no_outliers.index[i]
            map_delta_global = maps_xy[idx] - maps_xy[middle_frame_index]
            map_distance_global = calc_distance(map_delta_global, map_delta_global[60, 60, :])
            map_global = map_distance_global / map_unit
            map_global = offset_map_fov(map_global, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
            summary_global = calc_parametrics_global(map_global, map_fov, label, df_frame_no_outliers.loc[idx, 'frame_num'])
            summary.update(summary_global)
            plot_map_global(map_global, fname_str=label)
        except ValueError:
            logger.error('Error calculating and plotting global PS map')
            pass
            
    return summary


@dispatch(pd.DataFrame, dict, pd.DataFrame, list, list, np.ndarray, np.ndarray)
def eval_KPIs(df_frame, params, summary_df, maps_xy, maps_dxdy, middle_xy, middle_dxdy):
    def calc_distance(map_xy, ref0):
        #calculate distance map, relative to ref0, ref0 could be map of same shape, or x,y coor 1-dim array
        xi_full, yi_full, _ = map_xy.shape

        map_distance = np.empty((xi_full, yi_full))
        map_distance[:,:] = np.nan
        
        map_xy_delta = map_xy - ref0
        
        for i in range(xi_full):
            for j in range(yi_full):
                map_distance[i, j] = np.sqrt((map_xy_delta[i, j, 0] ** 2 + map_xy_delta[i, j, 1] ** 2))
        
        return map_distance

    map_dxdy_median = calc_median_map(maps_dxdy, min_num=1)
    assert(params['dxdy_spacing'] > 0)
    unitspacing_xy = map_dxdy_median[60, 60] / params['dxdy_spacing']
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx ** 2 + yy ** 2) + sys.float_info.epsilon # takes care of divide-by-zero warning
    
    summary = summary_df.to_dict(orient='records')[0]
    (xi_fov, yi_fov) = [summary['fov_dot_x'] - summary['center_dot_x'], summary['fov_dot_y'] - summary['center_dot_y']] / unitspacing_xy
    summary['version'] = cf.get_version()
    summary['xi_fov'] = xi_fov
    summary['yi_fov'] = yi_fov
    logger.info('Generated Maps are shifted by (%0.2f, %0.2f)', summary['map_x_shift'], summary['map_y_shift'])
    logger.info('FOV dot is indexed at (%0.2f, %0.2f)', xi_fov, yi_fov)
    
    # Local PS
    try:
        middle_map_dxdy_norm = middle_dxdy / map_dxdy_median # Normalize map_dxdy
        middle_map_dxdy_norm_fov = offset_map_fov(middle_map_dxdy_norm, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
        avg_map_dxdy = np.mean(maps_dxdy, axis=0)
        avg_map_dxdy_fov = offset_map_fov(avg_map_dxdy, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
        if params['filter_percent'] > 0:
            # Filter resultant map by removing filter_percent
            axis = ['X', 'Y']
            filtered_map = np.empty(np.shape(middle_map_dxdy_norm_fov))
            filtered_map.fill(np.nan)
            for j in range(len(axis)):
                mapj = middle_map_dxdy_norm_fov[:, :, j]
                upper = 1 + (params['filter_percent'] / 100)
                lower = 1 - (params['filter_percent'] / 100)
                resj = np.where(((mapj > upper) | (mapj < lower)) & (upper > lower), np.nan, mapj)
                filtered_map[:, :, j] = resj
                # if j == 0:
                #     df = pd.DataFrame(resj)
                #     df.to_csv(os.path.join(cf.output_path,'map_norm_dx_filtered.csv'))
                
            plot_map_norm(filtered_map, fname_str='Normalized_Map_')
            summary_local = calc_local_peripheral(filtered_map, map_fov, params)
            summary.update(summary_local) 
            summary_local = calc_local_central(avg_map_dxdy_fov, map_fov, params)
            #summary_local = calc_local_central(middle_map_dxdy_norm_fov, map_fov, params)
            summary.update(summary_local)
        else:
            plot_map_norm(middle_map_dxdy_norm_fov, fname_str='Normalized_Map_')
            summary_local = calc_local_peripheral(middle_map_dxdy_norm_fov, map_fov, params)
            summary.update(summary_local)
            summary_local = calc_local_central(avg_map_dxdy_fov, map_fov, params)
            #summary_local = calc_local_central(middle_map_dxdy_norm_fov, map_fov, params)
            summary.update(summary_local)
    except ValueError:
        logger.error('Error calculating and plotting local PS map')
        pass
        
    # Global PS
    map_distance = calc_distance(middle_xy, middle_xy[60, 60, :])
    map_unit = map_distance / map_fov + sys.float_info.epsilon # takes care of divide-by-zero warning
    df_frame_no_outliers = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0) & (df_frame['flag_slope_outlier'] == 0)] # filter out outlier frames
      
    for i in [0, -1]: #first and last frame
        if i == 0:
            label = 'Right Gaze'
        else:
            label = 'Left Gaze'
        try:
            idx = df_frame_no_outliers.index[i]
            map_delta_global = maps_xy[idx] - middle_xy
            map_distance_global = calc_distance(map_delta_global, map_delta_global[60, 60, :])
            map_global = map_distance_global / map_unit
            map_global = offset_map_fov(map_global, xi_fov, yi_fov, params['map_x_shift'], params['map_y_shift'])
            summary_global = calc_parametrics_global(map_global, map_fov, label, df_frame_no_outliers.loc[idx, 'frame_num'])
            summary.update(summary_global)
            plot_map_global(map_global, fname_str=label)
        except ValueError:
            logger.error('Error calculating and plotting global PS map')
            pass
            
    return summary
           

def find_middle_frame(df_frame, width, height):
    df_frame.set_index('index', inplace=True)   # use the index column as the new df_frame index 
    # determine center dot and FOV outliers
    median_center_x = np.median(df_frame['center_dot_x'])  # median of center dot locations, use this to determine center dot outlier
    median_center_y = np.median(df_frame['center_dot_y'])
    if (median_center_x > 2/3 * width) or (median_center_x < 1/3 * width) or (median_center_y > 2/3 * height) or (median_center_y < 1/3 * height): 
        raise Exception('Error: Center dot outside of ROI (middle 1/3)')
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
        df_frame.loc[i, 'dist_center_dot'] = np.sqrt((df_frame.loc[i,'center_dot_x'] - median_center_x) ** 2 + (df_frame.loc[i,'center_dot_y'] - median_center_y) ** 2)
        if df_frame.loc[i,'dist_center_dot'] > 50:
            df_frame.loc[i,'flag_center_dot_outlier']= 1
            num_outliers += 1
            logger.warning('Center dot outlier detected on frame #%s', df_frame.loc[i,'frame_num'])

        # determine if FOV dot is outlier, if d < 25px, mark as outlier, if y distance > 200px, outlier   
        df_frame.loc[i, 'dist_fov_center'] = np.sqrt((df_frame.loc[i,'fov_dot_x'] - df_frame.loc[i,'center_dot_x']) ** 2 + (df_frame.loc[i,'fov_dot_y'] - df_frame.loc[i,'center_dot_y']) ** 2)        
        if (df_frame.loc[i,'dist_fov_center'] < 25) or (np.abs(df_frame.loc[i,'fov_dot_y'] - df_frame.loc[i,'center_dot_y'])) > 200:
            df_frame.loc[i, 'flag_fov_dot_outlier'] = 1
            num_fov_outliers += 1
            logger.warning('FOV dot outlier detected on frame #%s', df_frame.loc[i,'frame_num'])
        
        # check slope    
        if (np.abs(df_frame.loc[i,'hor_slope']) > 0.1) or (np.abs(df_frame.loc[i,'ver_slope']) > 0.1):
            df_frame.loc[i, 'flag_slope_outlier'] = 1
            num_slope_outliers +=1
            logger.warning('Slope outlier detected on frame #%s', df_frame.loc[i,'frame_num'])

    df_frame['num_frames'] = len(df_frame.index)
    df_frame['num_center_dot_outlier'] = num_outliers
    df_frame['num_fov_dot_outlier'] = num_fov_outliers
    df_frame['num_slope_outlier'] = num_slope_outliers
    df_frame['num_total_outlier'] = num_outliers + num_fov_outliers + num_slope_outliers
    
    # find middle frame by min(d_fov_center)
    df_frame_no_outliers = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0) & (df_frame['flag_slope_outlier'] == 0)] # filter out outlier frames
    min_d_fov_center = np.min(df_frame_no_outliers['dist_fov_center'])
    middle_frame_index = df_frame.loc[df_frame['dist_fov_center'] == min_d_fov_center].index[0]
    
    return middle_frame_index
    

def find_outliers(df_frame, width, height):
    df_frame.set_index('index', inplace=True)   # use the index column as the new df_frame index 
    # determine center dot and FOV outliers
    median_center_x = np.median(df_frame['center_dot_x'])  # median of center dot locations, use this to determine center dot outlier
    median_center_y = np.median(df_frame['center_dot_y'])
    if (median_center_x > 2/3 * width) or (median_center_x < 1/3 * width) or (median_center_y > 2/3 * height) or (median_center_y < 1/3 * height): 
        raise Exception('Error: Center dot outside of ROI (middle 1/3)')
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
        df_frame.loc[i, 'dist_center_dot'] = np.sqrt((df_frame.loc[i,'center_dot_x'] - median_center_x) ** 2 + (df_frame.loc[i,'center_dot_y'] - median_center_y) ** 2)
        if df_frame.loc[i,'dist_center_dot'] > 50:
            df_frame.loc[i,'flag_center_dot_outlier']= 1
            num_outliers += 1
            logger.warning('Center dot outlier detected on frame #%s', df_frame.loc[i,'frame_num'])

        # determine if FOV dot is outlier, if d < 25px, mark as outlier, if y distance > 200px, outlier   
        df_frame.loc[i, 'dist_fov_center'] = np.sqrt((df_frame.loc[i,'fov_dot_x'] - df_frame.loc[i,'center_dot_x']) ** 2 + (df_frame.loc[i,'fov_dot_y'] - df_frame.loc[i,'center_dot_y']) ** 2)        
        if (df_frame.loc[i,'dist_fov_center'] < 25) or (np.abs(df_frame.loc[i,'fov_dot_y'] - df_frame.loc[i,'center_dot_y'])) > 200:
            df_frame.loc[i, 'flag_fov_dot_outlier'] = 1
            num_fov_outliers += 1
            logger.warning('FOV dot outlier detected on frame #%s', df_frame.loc[i,'frame_num'])
        
        # check slope    
        if (np.abs(df_frame.loc[i,'hor_slope']) > 0.1) or (np.abs(df_frame.loc[i,'ver_slope']) > 0.1):
            df_frame.loc[i, 'flag_slope_outlier'] = 1
            num_slope_outliers +=1
            logger.warning('Slope outlier detected on frame #%s', df_frame.loc[i,'frame_num'])

    df_frame['num_frames'] = len(df_frame.index)
    df_frame['num_center_dot_outlier'] = num_outliers
    df_frame['num_fov_dot_outlier'] = num_fov_outliers
    df_frame['num_slope_outlier'] = num_slope_outliers
    df_frame['num_total_outlier'] = num_outliers + num_fov_outliers + num_slope_outliers
    