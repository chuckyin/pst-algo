#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import config.config as cf

from config.logging import logger



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


def plot_map_norm(map_norm, fname_str=''):
    xi_full, yi_full, dim = map_norm.shape
    xi_range = int((xi_full - 1) / 2)
    yi_range = int((yi_full - 1) / 2)
    X, Y = np.meshgrid(np.linspace(-xi_range,xi_range,xi_full),np.linspace(-yi_range,yi_range,yi_full))
    
    for i in [0, 1]: #x,y
        if i == 0:
            axis = 'X'
        else:
            axis = 'Y'
        fig = plt.figure(figsize=(5, 5), dpi=200) 
        ax = fig.add_subplot(111)
        plt.title('Normalized ' + axis + '-Spacing Change')
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

    
def offset_map_fov (map_input, xi_fov, yi_fov):
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
    map_output = shift_fillnan(map_input, int(np.round(-xi_fov)), axis=0) #shift map relative to FOV 
    map_output = shift_fillnan(map_output, int(np.round(-yi_fov)), axis=1)
    
    return map_output

    
def calc_parametrics_local(map_local, map_fov):
    #calculate the parametrics based on the map
    summary = {}
    start_r = -1
    radii = [25, 35, 45, 60]  #need to start from small to large
    axis = ['x', 'y']
    th = [0.5, 1, 5]  #in percent
    
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


def eval_KPIs(df_frame, params, middle_frame_index, frame_nums, maps_xy, maps_dxdy):
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
    
    summary = df_frame.loc[df_frame['dist_fov_center'] == np.min(df_frame['dist_fov_center'])].to_dict(orient='records')[0]
    (xi_fov, yi_fov) = [summary['fov_dot_x'] - summary['center_dot_x'], summary['fov_dot_y'] - summary['center_dot_y']] / unitspacing_xy
    summary['version'] = cf.get_version()
    summary['xi_fov'] = xi_fov
    summary['yi_fov'] = yi_fov
    logger.info('Middle frame determined to be #%s', summary['frame_num'])
    logger.info('FOV dot is indexed at (%0.2f, %0.2f)', xi_fov, yi_fov)
    
    # Local PS
    try:
        map_dxdy_norm = maps_dxdy[middle_frame_index] / map_dxdy_median # Normalize map_dxdy
        map_dxdy_norm_fov = offset_map_fov(map_dxdy_norm, xi_fov, yi_fov)
        summary_local = calc_parametrics_local(map_dxdy_norm_fov, map_fov)
        summary.update(summary_local) 
        plot_map_norm(map_dxdy_norm_fov)
    except ValueError:
        logger.error('Error calculating and plotting local PS map')
        pass
    
    # Global PS
    map_distance = calc_distance(maps_xy[middle_frame_index], maps_xy[middle_frame_index][60, 60, :])
    map_unit = map_distance / map_fov + sys.float_info.epsilon # takes care of divide-by-zero warning
      
    for i in [0, -1]: #first and last frame
        if i == 0:
            label = 'Right Gaze'
        else:
            label = 'Left Gaze'
        try:
            idx = df_frame.index[i]
            map_delta_global = maps_xy[idx] - maps_xy[middle_frame_index]
            map_distance_global = calc_distance(map_delta_global, map_delta_global[60, 60, :])
            map_global = map_distance_global / map_unit
            map_global = offset_map_fov(map_global, xi_fov, yi_fov)
            summary_global = calc_parametrics_global (map_global, map_fov, label, frame_nums)
            summary.update(summary_global)
            plot_map_global(map_global, fname_str=label)
        except ValueError:
            logger.error('Error calculating and plotting global PS map')
            pass
            
    return summary
           

def find_middle_frame(df_frame, width, height):
    df_frame.set_index('index', inplace=True)   # use the index column as the new df_frame index 
    #determine center dot and FOV outliers
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
    
    #find middle frame by min(d_fov_center)
    df_frame = df_frame[(df_frame['flag_center_dot_outlier'] == 0) & (df_frame['flag_fov_dot_outlier'] == 0) & (df_frame['flag_slope_outlier'] == 0)] # filter out outlier frames
    min_d_fov_center = np.min(df_frame['dist_fov_center'])
    middle_frame_index = df_frame.loc[df_frame['dist_fov_center'] == min_d_fov_center].index[0]
    
    return middle_frame_index
