#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:27:06 2022

@author: melshaer0612

# Version Complete -- 01/25/2023

"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import re
import time

from datetime import timedelta
from skimage.transform import radon
from scipy.interpolate import griddata
import csv


class Frame:
    def __init__(self):
        self.dots = []
        self.center_dot = None
        self.med_dot_size = None
        self.med_dot_dist = None
        self.hor_slope = None
        self.ver_slope = None
        self.hor_lines = None
        self.ver_lines = None
        self.dotsxy_indexed = {}  # key(x, y) : value(xi, yi)
        self.xi_range =60
        self.yi_range =60
        self.map_xy = np.empty((self.xi_range*2+1,self.yi_range*2+1,2)) # init map w +/-60 deg 
        self.map_xy[:,:,:] =np.nan
        self.map_dxdy = np.empty((self.xi_range*2+1,self.yi_range*2+1,2))
        self.map_dxdy[:,:,:] =np.nan
   
        
    def get_x_y(self, inv=False):
        if inv:
            y = np.asarray([dot.x for dot in self.dots])
            x = np.asarray([dot.y for dot in self.dots])  
        else:
            x = np.asarray([dot.x for dot in self.dots])
            y = np.asarray([dot.y for dot in self.dots])
            
        return x, y
    
       
    def calc_dot_size_dist(self):
        x, y = self.get_x_y()        
        sq_dist = np.gradient(x) ** 2 + np.gradient(y) ** 2
        
        self.med_dot_dist = np.median(np.sqrt(sq_dist))
        self.med_dot_size = np.median([dot.size for dot in self.dots])
        
        return self.med_dot_size, self.med_dot_dist
    
    
    def get_slopes(self, init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error):
        def calc_slopes(x, y, slope, error):    
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            index_mid_dot = np.argsort(np.sqrt((mean_x - x) ** 2
                                               + (mean_y - y) ** 2))[0]
            used_dot_x = x[index_mid_dot]
            used_dot_y = y[index_mid_dot]
            list_tmp = np.sqrt(np.ones(len(self.dots), dtype=np.float32) * slope ** 2 + 1.0)
            list_tmp2 = used_dot_y * np.ones(len(self.dots), dtype=np.float32) \
                        - slope * used_dot_x
            list_dist = np.abs(slope * x - y + list_tmp2) / list_tmp
            dots_selected = [dot for i, dot in enumerate(self.dots) if list_dist[i] < error]
            dots_selected_x = np.asarray([dot.x for dot in dots_selected])
            dots_selected_y = np.asarray([dot.y for dot in dots_selected])
            if len(dots_selected) > 1:
                (_slope, _) = np.polyfit(dots_selected_x, dots_selected_y, 1)
            else:
                _slope = slope
             
            return _slope
        
        x, y = self.get_x_y()
        self.hor_slope = calc_slopes(-x, -y, init_hor_slope, hor_dist_error) 
        self.ver_slope = calc_slopes(x, y, init_ver_slope, ver_dist_error)
        
        return self.hor_slope, self.ver_slope
    
    
    def check_dot_on_line(self, dot1, dot2, ratio, num_dots_miss, ver):
        if ver:
            slope = self.ver_slope 
        else:
            slope = self.hor_slope
        check = False
        dist_error = ratio * self.med_dot_dist
        search_dist = num_dots_miss * self.med_dot_dist
        xmin = dot1[1] - search_dist
        xmax = dot1[1] + search_dist
        if xmin < dot2[1] < xmax:
            ntemp1 = np.sqrt(slope * slope + 1.0)
            ntemp2 = dot1[0] - slope * dot1[1]
            dist_d12 = np.abs(slope * dot2[1] - dot2[0] + ntemp2) / ntemp1
            if dist_d12 < dist_error:
                check = True
                
        return check

        
    def group_lines(self, ratio=0.2, num_dots_miss=6, accepted_ratio=0.3):
        def get_lines(x, y, slope, ver=False):
            list_dots_left = np.vstack((y, x)).T
            if ver:
                list_dots_left = np.fliplr(list_dots_left)
            list_dots_left = list_dots_left[list_dots_left[:, 1].argsort()]
            num_dots_left = len(list_dots_left)
            list_lines = []
            while num_dots_left > 1:
                dot1 = list_dots_left[0]
                dots_selected = np.asarray([dot1])
                pos_get = [0]
                for i in range(1, len(list_dots_left)):
                    dot2 = list_dots_left[i]
                    check = self.check_dot_on_line(dot1, dot2, ratio, num_dots_miss, ver)
                    if check:
                        dot1 = dot2
                        dots_selected = np.vstack((dots_selected, dot2))
                        pos_get.append(i)
                list_pos = np.arange(0, len(list_dots_left), dtype=np.int32)
                pos_get = np.asarray(pos_get, dtype=np.int32)
                list_pos_left = np.asarray(
                    [pos for pos in list_pos if pos not in pos_get], dtype=np.int32)
                list_dots_left = list_dots_left[list_pos_left]
                num_dots_left = len(list_dots_left)
                if len(dots_selected) > 1:
                    if ver:
                        list_lines.append(dots_selected)
                    else:
                        dots_selected = np.fliplr(dots_selected)
                        list_lines.append(dots_selected)
            list_len = [len(i) for i in list_lines]
            len_accepted = np.int16(accepted_ratio * np.max(list_len))
            lines_selected = [line for line in list_lines if len(line) > len_accepted] 
            if ver:
                lines_selected = sorted(lines_selected, key=lambda list_: np.mean(list_[:, 0]))
            else: 
                lines_selected = sorted(lines_selected, key=lambda list_: np.mean(list_[:, 1]))
            
            return lines_selected
        
        x, y = self.get_x_y()
        hor_lines = get_lines(x, y, self.hor_slope)
        #print(len(hor_lines))
        self.hor_lines = [list(map(tuple, line)) for line in hor_lines]
        
        x, y = self.get_x_y()
        ver_lines = get_lines(x, y, self.ver_slope, ver=True)
        #print(len(ver_lines))
        self.ver_lines = [list(map(tuple, line)) for line in ver_lines]
        
        return self.hor_lines, self.ver_lines
    
    
    def find_index(self):
        # The indexed dots are those grouped on the horizontal and vertical lines simultanoeusly.
        all_hor_pts = np.concatenate(self.hor_lines, axis=0).tolist() # flatten
        all_ver_pts = np.concatenate(self.ver_lines, axis=0).tolist() # flatten
        
        common_pts = [tuple(pt) for pt in all_ver_pts if pt in all_hor_pts]
        
        # Register all indices to the center dot
        center_dot_pt = (self.center_dot.x, self.center_dot.y)
        try:
            common_pts.index(center_dot_pt)   
            center_dot_hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if center_dot_pt in line])
            center_dot_vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if center_dot_pt in line])
        except ValueError:
            print('Warning: Center Dot was not indexed. Will use the closest dot instead.')
            dist_2 = np.sum((np.asarray(common_pts) - np.asarray(center_dot_pt)) ** 2, axis=1)
            pseudo_center_pt = common_pts[np.argmin(dist_2)]
            center_dot_hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pseudo_center_pt in line])
            center_dot_vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pseudo_center_pt in line])
            return [],[]
        
        for pt in common_pts:
            hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pt in line])
            vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pt in line])
            self.dotsxy_indexed[pt] = list(zip(vi - center_dot_vi, hi - center_dot_hi))
        
        return vi - center_dot_vi , hi - center_dot_hi
            
    def generate_map_xy(self):
        # from indexed dots to generate xy coordinate map
        cnt =0
        for coord in self.dotsxy_indexed:
            ind = self.dotsxy_indexed[coord][0]

            if np.abs(ind[0]) <= self.xi_range:
                if np.abs(ind[1]) <= self.yi_range:
                    self.map_xy[ind[0]+ self.xi_range,ind[1]+self.yi_range,:] = coord #offset center
                    cnt+=1
        print('Map generation completed with ' , str(cnt), ' indexed dots' )  

    def generate_map_dxdy(self,spacing:int):
        # from map_xy generate derivative map, spacing as int
        pos = int(np.round(spacing/2))
        neg = pos - spacing
        
        map_xy_copy = np.copy(self.map_xy)
        mapx = map_xy_copy[:,:,0]
        mapdx = shift_fillnan(mapx,-pos,0) - shift_fillnan(mapx,-neg,0)
        mapy = map_xy_copy[:,:,1]
        mapdy = shift_fillnan(mapy,-pos,1) - shift_fillnan(mapy,-neg,1)
        
        self.map_dxdy[:,:,0]=mapdx
        self.map_dxdy[:,:,1]=mapdy
            
    def draw_lines_on_image(self, image, width, height, filename):
       image_cpy = image.copy()
       for line in self.hor_lines:
           line_int = [(int(pt[0]), int(pt[1])) for pt in line]
           for pt1, pt2 in zip(line_int, line_int[1:]): 
               cv2.line(image_cpy, pt1, pt2, [255, 0, 0], 2)
       for line in self.ver_lines:
           line_int = [(int(pt[0]), int(pt[1])) for pt in line]
           for pt1, pt2 in zip(line_int, line_int[1:]): 
               cv2.line(image_cpy, pt1, pt2, [0, 255, 0], 2)
       cv2.imwrite(os.path.join(output_path, filename), image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])
        

                    
class Dot:            
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
            
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
        
    
current_path = os.getcwd()

# input_path = os.path.join(current_path, '60deg_test_55p5' + '/')
# output_path = os.path.join(current_path, '60deg_test_55p5_output' + '/')
# image_files = glob.glob(input_path + '*.tiff')

image_files = ['test4.tiff']
input_path = current_path
output_path = current_path
driver = 'FATP'
dxdy_spacing = 4


#----Xuan test------

current_path = r'/Users/xuawang/Dropbox (Meta)/Pupil Swim Metrology/PST/20221121 2d dots 60deg/Arcata EVT2 A/test2_55p5/'

input_path = current_path
output_path = os.path.join(current_path, 'output' + '/')
if not os.path.exists(output_path):
    os.makedirs(output_path)

image_files = glob.glob(input_path + '*.tiff')
image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
#image_files =image_files[:1]
image_files =image_files[20:120:20]  #every nth image

#-------------

log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.txt'
csv_file = os.path.join(output_path, 'dots.csv')
csv_file_frame = os.path.join(output_path, 'frames.csv')
csv_file_summary = os.path.join(output_path, 'summary.csv')

# dot_params = {'min_dist' : 3,
#               'min_threshold' : 9, 
#               'max_threshold' : 240,
#               'threshold_step' : 3, 
#               'min_area' : 5, 
#               'max_area' : 2000, 
#               'invert' : True, 
#               'subtract_median' : False,
#               'filter_byconvexity' : False,
#               'min_convexity': 0.6, 
#               'filter_bycircularity' : True}

dot_params = {"min_dist": 3,
              "min_threshold":1,
              "max_threshold": 1000,
              "threshold_step":2, 
              "min_area":50, 
              "max_area": 2000, 
              "invert" : True, 
              "subtract_median": False,
              "filter_byconvexity": True, 
              "min_convexity": 0.75, 
              "filter_bycircularity": False}


def write_log(*args, filename):
    log_file = open(os.path.join(output_path, filename), 'a+')
    line = ' '.join([str(a) for a in args])
    log_file.write(line + '\n')
    print(line)
    

def distance_kps(dot1, dot2):
    return np.sqrt((dot1.x - dot2.x) ** 2 + (dot1.y - dot2.y) ** 2)


def detect_blobs(params, image):
    if params['invert']:
        image = 255 - image

    if params['subtract_median']:
        med_blur = cv2.medianBlur(image, 11)
        image = 255 - 255 / (np.max(np.abs(med_blur - image))) * np.abs(med_blur - image)
        
    # Setup SimpleBlobDetector parameters.
    blob_params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    blob_params.minThreshold = int(params['min_threshold'])
    blob_params.maxThreshold = int(params['max_threshold'])
    blob_params.thresholdStep = int(params['threshold_step'])
    blob_params.minDistBetweenBlobs = int(params['min_dist'])
    # Filter by Area
    blob_params.filterByArea = True
    blob_params.minArea = int(params['min_area'])
    blob_params.maxArea = int(params['max_area'])
    # Filter by Circularity
    blob_params.filterByCircularity =params['filter_bycircularity']
    blob_params.filterByInertia = False
    blob_params.filterByConvexity = params['filter_byconvexity']
    blob_params.minConvexity = params['min_convexity']   # 0.9 filters out most noise at center, lost a few at edge
    
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(image.astype('uint8'))
    x = [kp.pt[0] for kp in keypoints]
    y = [kp.pt[1] for kp in keypoints]
    size = [kp.size for kp in keypoints]

    return x, y, size


def find_center_dot(dots, height, width):
    # Find center dot by size, within the ROI (this is to avoid large blob at other areas)
    max_size = 0
    max_size_index = 0
    center_roi = 1/3 #find the center spot within the center of the picture only
    for kp in range(len(dots)):
        if dots[kp].x > (0.5-0.5* center_roi)*width and dots[kp].x < (0.5+0.5* center_roi)*width:  #x filter
            if dots[kp].y > (0.5-0.5* center_roi)*height and dots[kp].y < (0.5+0.5* center_roi)*height: #y filter
                if dots[kp].size > max_size:
                    max_size = dots[kp].size
                    max_size_index = kp
    center_dot = Dot(dots[max_size_index].x, dots[max_size_index].y, dots[max_size_index].size) # creates a new object
    
    return center_dot
    
        
def find_dots(image):
    image = prep_image(image, normalize_and_filter=False, binarize=False)
    x, y, size = detect_blobs(dot_params, image)
    frame = Frame()
    frame.dots = [Dot(x_i, y_i, size_i) for (x_i, y_i, size_i) in list(zip(x, y, size))]
    
    return frame
    
     
def find_fov(image, height, width):
    roi_hei = np.int16(height / 3)
    roi_image = image[roi_hei:height - roi_hei]
    image = prep_image(roi_image, normalize_and_filter=True, binarize=True)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        outer_c_index = int(np.where((hierarchy[0,:,2] != -1) & (hierarchy[0,:,3] == -1))[0])
        inner_c_index = int(np.where((hierarchy[0,:,2] == -1) & (hierarchy[0,:,3] != -1))[0])
        M_inner = cv2.moments(contours[inner_c_index])
        M_outer = cv2.moments(contours[outer_c_index])
    except TypeError:
        return None
    
    fov_x = int(M_inner['m10'] / M_inner['m00'])
    fov_y = int(M_inner['m01'] / M_inner['m00']) + roi_hei
    fov_size = M_outer['m00'] / M_inner['m00']
    fov_dot = Dot(fov_x, fov_y, fov_size)
        
    return fov_dot


def prep_image(image, normalize_and_filter, binarize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if normalize_and_filter:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 3, 100, 100)
    if binarize:
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # Could add cv2.THRESH_OTSU
        return thresh
    else:
        return gray   
    

def get_initial_slopes(image, height, width, ratio):
    # Select a part of image for slope calculations
    ratio = np.clip(ratio, 0.05, 1.0)
    depad_hei = np.int16((height - ratio * height) / 2)
    depad_wid = np.int16((width - ratio * width) / 2)
    roi_image = image[depad_hei:height - depad_hei, depad_wid:width - depad_wid]
    roi_height, roi_width = roi_image.shape
    
    radi = np.pi / 180.0
    coarse_range = 30.0  # Degree
    
    # Horizontal Slope
    hor_list_angle = 90.0 + np.arange(-coarse_range, coarse_range + 1.0)
    hor_projections = radon(np.float32(roi_image), theta=hor_list_angle, circle=False)
    hor_list_max = np.amax(hor_projections, axis=0)
    hor_best_angle = -(hor_list_angle[np.argmax(hor_list_max)] - 90.0)
    hor_dist_error = 0.5 * roi_width * np.tan(radi) / np.cos(hor_best_angle * radi)
    init_hor_slope = np.tan(hor_best_angle * radi)
    
    # Vertical Slope
    ver_list_angle = np.arange(-coarse_range, coarse_range + 1.0)
    ver_projections = radon(np.float32(roi_image), theta=ver_list_angle, circle=False)
    ver_list_max = np.amax(ver_projections, axis=0)
    ver_best_angle = (ver_list_angle[np.argmax(ver_list_max)])
    ver_dist_error = 0.5 * roi_width * np.tan(radi) / np.cos(ver_best_angle * radi)
    init_ver_slope = np.tan(ver_best_angle * radi)
    
    return init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error


def draw_dots(image, dots, filename):
    image_cpy = image.copy()
    for dot in dots:
        if len(dots) > 1:
            cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size/2), (0, 255, 0), 3) # Green Dots
        else:
            cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size/2), (0, 0, 255), 3) # Red Circle    
    cv2.imwrite(os.path.join(output_path, filename), image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])


def process_csv(filename):
    df = pd.read_csv(filename)
    df_sub = df[['x', 'y', 'xi', 'yi', 'frame_num']]
    df_sub = df_sub.rename(columns={'x':'x_temp', 'y':'y_temp'})

    df['xi1'] = df['xi'] - 1
    df['yi1'] = df['yi'] - 1

    xi = df_sub['xi'].values
    yi = df_sub['yi'].values
    frame_num = df_sub['frame_num'].values
    v = np.vstack([xi, yi, frame_num]).T.tolist()
    vt = [tuple(vi) for vi in v]
    df_sub['key'] = vt

    xi1 = df['xi1'].values
    yi1 = df['yi1'].values
    xi_df = df['xi'].values
    yi_df = df['yi'].values
    frame_num_df = df['frame_num'].values
    vx_df = np.vstack([xi1, yi_df, frame_num_df]).T.tolist()
    vy_df = np.vstack([xi_df, yi1, frame_num_df]).T.tolist()
    vxt_df = [tuple(vi) for vi in vx_df]
    vyt_df = [tuple(vi) for vi in vy_df]
    df['keyx'] = vxt_df
    df['keyy'] = vyt_df

    merged_df_x = pd.merge(df, df_sub[['x_temp', 'key']], left_on=['keyx'], right_on=['key'], how='left').drop(columns=['keyx', 'key'])
    merged_df = pd.merge(merged_df_x, df_sub[['y_temp', 'key']], left_on=['keyy'], right_on=['key'], how='left').drop(columns=['keyy', 'key'])

    merged_df['dx'] = merged_df['x'] - merged_df['x_temp']
    merged_df['dy'] = merged_df['y'] - merged_df['y_temp']

    df2 = pd.DataFrame()
    df2['dx_med'] = merged_df.groupby(['xi', 'yi'])['dx'].median()
    df2['dy_med'] = merged_df.groupby(['xi', 'yi'])['dy'].median()

    df2.reset_index(inplace=True)

    v_df2 = np.vstack([df2['xi'].values, df2['yi'].values]).T.tolist()
    v_merged = np.vstack([merged_df['xi'].values, merged_df['yi'].values]).T.tolist()
    vt_df2 = [tuple(vi) for vi in v_df2]
    vt_merged = [tuple(vi) for vi in v_merged]
    df2['key'] = vt_df2
    merged_df['key'] = vt_merged

    merged_df2 = pd.merge(merged_df, df2[['dx_med', 'dy_med', 'key']], on=['key'], how='left').drop(columns=['key'])

    merged_df2['dx_norm'] = merged_df2['dx'] / merged_df2['dx_med']
    merged_df2['dy_norm'] = merged_df2['dy'] / merged_df2['dy_med']
    
    # ------------------------------?
    # Plots
    y = merged_df2['yi'].tolist()
    x = merged_df2['xi'].tolist()
    dx = merged_df2['dx_norm'].tolist()
    dy = merged_df2['dy_norm'].tolist()
    
    # Suggestion 1: Hexbin Plots
    plt.hexbin(x, y, dx, cmap=cm.jet)
    plt.colorbar()
    plt.title('dx')
    plt.figure()
    plt.hexbin(x, y, dy, cmap=cm.jet)
    plt.colorbar()
    plt.title('dy')

    # Suggestion 2: Scatter Plots
    plt.scatter(x, y, c=dx, cmap=cm.jet)
    plt.colorbar()
    plt.figure()
    plt.scatter(x, y, c=dy, cmap=cm.jet)
    plt.colorbar()
    
    # Suggestion 3: Joint Distribution Plots
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "hex", data = merged_df2['dx'])
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "kde", data = merged_df2['dx'])
    sns.jointplot(x = merged_df2['xi'], y = merged_df2['yi'], kind = "scatter", data = merged_df2['dx'])
    
    # Suggestion 4: Contour Plots
    data = np.array([x, y, dx]).T
    X, Y = np.meshgrid(data[:,0], data[:,1])
    Z = griddata((data[:,0], data[:,1]), data[:,2], (X, Y), method='nearest')
    plt.contourf(X, Y, Z)
    
    # Or
    levels = 0.5
    plt.contour(X, Y, Z, levels=levels)
    
    # Or
    merged_na = merged_df2.dropna()
    y = merged_na['yi'].tolist()
    x = merged_na['xi'].tolist()
    dx = merged_na['dx_norm'].tolist()
    dy = merged_na['dy_norm'].tolist()
    plt.tricontourf(x, y, dx)

def shift_fillnan (arr, shift, axis):
    #shift 2d array with nan fillin
    arr = np.roll(arr, shift, axis)
    if axis ==0:
        if shift>=0:
            arr[:shift,:] =np.nan
        else:
            arr[shift:,:] =np.nan
    elif axis ==1:
        if shift>=0:
            arr[:,:shift] =np.nan
        else:    
            arr[:,shift:] =np.nan
    else:
        pass
    
    return arr    

def calc_median_map(maps, min_num =3):
    #calculate the median map of a maps (list), with min number of valid elements (will output nan if <min_num)
    cnt =0
    maps_arr = np.asarray(maps)
    l,xi,yi,dim = np.shape(maps_arr)
    output = np.empty((xi,yi,dim))
    if min_num > l:
        min_num = l
        print ("Warning: min number of valid elements reduce to: ", str(l))
    
    for i in range(xi):
        for j in range(yi):
            for k in range(dim):
                if np.count_nonzero(~np.isnan(maps_arr[:,i,j,k]))>=min_num:
                    output[i,j,k] = np.nanmedian(maps_arr[:,i,j,k])
                    cnt +=1
                else:
                    output[i,j,k] = np.nan
    print ('Median map calculateion completed with ' , str(cnt), ' dots' )                
    return output

# def calc_unit_spacing(map_xy_offset):
#     #calculate unit spacing at each location, used to calibrate the spacing from px to degree 
#     xi_full, yi_full, dim = map_xy_offset.shape
#     xi_range = int((xi_full -1)/2)
#     yi_range = int((yi_full -1)/2)    

#     map_unit = np.empty((xi_full,yi_full))
#     map_unit[:,:] =np.nan
    
#     for i in range(xi_full):
#         for j in range(yi_full):
#             map_unit[i,j] = np.sqrt((map_xy_offset[i,j,0]**2 + map_xy_offset[i,j,1]**2) / ((i-xi_range)**2 + (j-yi_range)**2))
    
#     return map_unit

def calc_distance(map_xy, ref0):
    #calculate distance map, relative to ref0, ref0 could be map of same shape, or x,y coor 1-dim array
    xi_full, yi_full, _ = map_xy.shape

    map_distance = np.empty((xi_full,yi_full))
    map_distance[:,:] =np.nan
    
    map_xy_delta = map_xy - ref0
    
    for i in range(xi_full):
        for j in range(yi_full):
            map_distance[i,j] = np.sqrt((map_xy_delta[i,j,0]**2 + map_xy_delta[i,j,1]**2))
    
    return map_distance

def plot_map_norm(map_norm, fname_str=''):
    
    xi_full, yi_full, dim = map_norm.shape
    xi_range = int((xi_full -1)/2)
    yi_range = int((yi_full -1)/2)
    X,Y =np.meshgrid(np.linspace(-xi_range,xi_range,xi_full),np.linspace(-yi_range,yi_range,yi_full))
    #map_norm = np.ma.array(map_norm, mask=np.isnan(map_norm))
    
    for i in [0,1]: #x,y
        if i ==0:
            axis = "x"
        else:
            axis = "y"
        fig=plt.figure(figsize=(5, 5),dpi=200) #figsize=(5, 5),
        ax = fig.add_subplot(111)
        plt.title('Normalized '+ axis + '-Spacing Change')
        plt.xlabel('xi')
        plt.ylabel('yi')
        levels = np.linspace(0.95, 1.05, 11)
        plt.contourf(X,Y,map_norm[:,:,i].T,levels=levels, cmap ='seismic',extend ='both')  #cmap ='coolwarm','bwr'
        ax.set_aspect('equal')
    
        plt.xlim(-xi_range, xi_range)
        plt.ylim(yi_range, -yi_range)
        
        radii = [25,45]
    
        for radius in radii:    
            theta = np.linspace(0, 2*np.pi, 100)
            a = radius*np.cos(theta)
            b = radius*np.sin(theta)
            plt.plot(a, b,color='green', linestyle=(0, (5, 5)), linewidth=0.4)  #'loosely dotted'
            
        plt.grid(color='grey', linestyle='dashed', linewidth=0.2)
        plt.rcParams["font.size"] = "5"
        plt.colorbar()
        plt.savefig(os.path.join(output_path, 'Normalized_Map_'+fname_str + axis +'.png'))
        #plt.show()
 
def plot_map_global(map_global, fname_str=''):
    
    xi_full, yi_full = map_global.shape
    xi_range = int((xi_full -1)/2)
    yi_range = int((yi_full -1)/2)
    X,Y =np.meshgrid(np.linspace(-xi_range,xi_range,xi_full),np.linspace(-yi_range,yi_range,yi_full))
    #map_global = np.ma.array(map_global, mask=np.isnan(map_global))

    fig=plt.figure(figsize=(5, 5),dpi=200) #figsize=(5, 5),
    ax = fig.add_subplot(111)
    plt.title('Global PS Map '+fname_str+' [degree]')
    plt.xlabel('xi')
    plt.ylabel('yi')
    levels = np.linspace(0, 1.0, 6)
    plt.contourf(X,Y,map_global[:,:].T,levels=levels, cmap ='coolwarm',extend ='max')  #cmap ='coolwarm','bwr'
    ax.set_aspect('equal')

    plt.xlim(-xi_range, xi_range)
    plt.ylim(yi_range, -yi_range)
    
    radii = [25,45]

    for radius in radii:    
        theta = np.linspace(0, 2*np.pi, 100)
        a = radius*np.cos(theta)
        b = radius*np.sin(theta)
        plt.plot(a, b,color='green', linestyle=(0, (5, 5)), linewidth=0.4)  #'loosely dotted'
        
    plt.grid(color='grey', linestyle='dashed', linewidth=0.2)
    plt.rcParams["font.size"] = "5"
    plt.colorbar()
    plt.savefig(os.path.join(output_path, 'Global_PS_Map_'+fname_str +'.png'))       
 
def offset_map_fov (map_input, xi_fov,yi_fov):
    #offset map and create FOV map (center of display)
    map_output = shift_fillnan (map_input, int(np.round(-xi_fov)), axis=0) #shift map relative to FOV 
    map_output = shift_fillnan (map_output, int(np.round(-yi_fov)), axis=1)
    
    return map_output
    
def calc_parametrics_local (map_local, map_fov):
    #calculate the parametrics based on the map
    summary ={}
    #zones =[]
    start_r =-1
    radii = [25,35,45,60]  #need to start from small to large
    axis = ['x','y']
    th = [1,5]  #in percent
    
    #define zones
    for i in range(len(radii)):
        zone =  (map_fov > start_r) * (map_fov <= radii[i])
        #zones.append(zone)
        start_r = radii[i]
        
        mapp_both_x =[]
        for j in range(len(axis)):
            
            mapp = map_local[:,:,j]
            summary['local_areatotal_d'+axis[j]+'_'+str(radii[i])]= np.count_nonzero(~np.isnan(mapp[zone]))
            summary['local_max_d'+axis[j]+'_'+str(radii[i])]= (np.nanmax(mapp[zone])-1)*100
            summary['local_min_d'+axis[j]+'_'+str(radii[i])]= (np.nanmin(mapp[zone])-1)*100
            summary['local_pct99_d'+axis[j]+'_'+str(radii[i])]= (np.nanpercentile(mapp[zone],99) -1)*100
            summary['local_pct1_d'+axis[j]+'_'+str(radii[i])]= (np.nanpercentile(mapp[zone],1) -1)*100
            summary['local_rms_d'+axis[j]+'_'+str(radii[i])]= (np.nanstd(mapp[zone]))*100
            
            
            for k in range(len(th)): 
                mapp_pos = (mapp>=1+th[k]/100) * zone
                mapp_neg = (mapp<=1-th[k]/100) * zone
                mapp_both = ((mapp>=1+th[k]/100) + (mapp<=1-th[k]/100))* zone
                summary['local_area_d'+axis[j]+'_th'+str(th[k])+'pctpos_'+str(radii[i])]= np.count_nonzero(mapp_pos)
                summary['local_area_d'+axis[j]+'_th'+str(th[k])+'pctneg_'+str(radii[i])]= np.count_nonzero(mapp_neg)
                summary['local_area_d'+axis[j]+'_th'+str(th[k])+'pct_'+str(radii[i])]= np.count_nonzero(mapp_both)
                if j ==0: # save x maps
                    mapp_both_x.append(mapp_both)
                elif j==1: 
                    mapp_both_combine = mapp_both + mapp_both_x[k]
                    summary['local_area_combined_th'+str(th[k])+'pct_'+str(radii[i])]= np.count_nonzero(mapp_both_combine)
                
    return summary


def calc_parametrics_global (map_global, map_fov, label, frame_num):
    #calculate the parametrics based on the map    
    summary ={}
    start_r =-1
    radii = [25,35,45,60]  #need to start from small to large
    #axis = ['x','y']
    #th = [1,5]  #in percent
    summary['global_'+label+'_frame_num']= frame_num
    
    #define zones
    for i in range(len(radii)):
        zone =  (map_fov > start_r) * (map_fov <= radii[i])
        start_r = radii[i]
        summary['global_'+label+'_max_'+str(radii[i])]= np.nanmax(map_global[zone])
        summary['global_'+label+'_pct99_'+str(radii[i])]= np.nanpercentile(map_global[zone],99)
        summary['global_'+label+'_median_'+str(radii[i])]= np.nanpercentile(map_global[zone],50)

    return summary
    
if __name__ == '__main__':
    start_time = time.monotonic()
    df = pd.DataFrame() #raw data, 1 blob per line
    df_frame = pd.DataFrame() #frame summary data, 1 frame per line
    cnt = 0
    maps_xy =[]
    maps_dxdy =[]
    frame_nums =[]
    #frames =[]
    for image_file in image_files:
        frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
        frame_nums.append(frame_num)
        df_frame_mini = pd.DataFrame({'frame_num':[frame_num],'index':[cnt]})
        #df_frame_mini['frame_num'] = frame_num
        image = cv2.imread(os.path.join(current_path, image_file))
        
        if driver =="MODEL":
            #image = np.fliplr(image)
            pass
        elif driver == "FATP": #rotate 180 deg as FATP is mounted upside down
            image =np.rot90(image,2)  

        print('Frame', frame_num, ': Processing started')
        height, width, _ = image.shape
        
        
        # ------Blob Detection --------------
        frame = find_dots(image)
        print('Frame', frame_num, ': Finding dots is complete, found ', str(len(frame.dots)), 'dots')
        df_frame_mini['total_dots'] = len(frame.dots)
        
        frame.center_dot = find_center_dot(frame.dots, height, width)
        print('Frame', frame_num, ': Center dot was found at ', frame.center_dot.__str__())
        df_frame_mini['center_dot_x'] = frame.center_dot.x
        df_frame_mini['center_dot_y'] = frame.center_dot.y
        
        fov_dot = find_fov(image, height, width)
        print('Frame', frame_num, ': Finding fov dot is complete')
        if fov_dot is None:
            write_log('ERROR: Could not find FOV point for frame ', str(frame_num), filename=log_file)          
        else:
            draw_dots(image, [fov_dot], filename=frame_num+'_fov.jpeg') # For debugging FOV dot detection
        
        draw_dots(image, frame.dots, filename=frame_num+'_dots.jpeg') # For debugging blob detection
        df_frame_mini['fov_dot_x'] = fov_dot.x
        df_frame_mini['fov_dot_y'] = fov_dot.y       
        
        
        #--------Group, indexing ----------------
        med_size, med_dist = frame.calc_dot_size_dist()
        print('Dot Size:', med_size, 'Distance:', med_dist)
        df_frame_mini['median_dot_size'] = med_size
        df_frame_mini['median_dot_spacing'] = med_dist
        
        print('Starting slope calculations for frame', frame_num)
        proc = prep_image(image, normalize_and_filter=True, binarize=False)
        init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = get_initial_slopes(proc, height, width, ratio=0.3)
        hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
        print('HSlope:', hor_slope, 'VSlope:', ver_slope)
        df_frame_mini['hor_slope'] = hor_slope
        df_frame_mini['ver_slope'] = ver_slope        
        
        hor_lines, ver_lines = frame.group_lines()
        frame.draw_lines_on_image(image, width, height, filename=frame_num+'_grouped.jpeg')
        vi, hi = frame.find_index()
        print('Finished indexing calculations for frame', frame_num)
        if vi != []:
            df_frame_mini['yi_max'] = np.max(vi)
            df_frame_mini['yi_min'] = np.min(vi)
        if hi != []:
            df_frame_mini['xi_max'] = np.max(vi)
            df_frame_mini['xi_min'] = np.min(vi)
        
        # generate maps
        frame.generate_map_xy()
        maps_xy.append(frame.map_xy)
        frame.generate_map_dxdy(dxdy_spacing)  #4x spacing
        df_frame_mini['dxdy_spacing'] = dxdy_spacing
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
    if median_center_x > 2/3* width or median_center_x < 1/3* width or median_center_y > 2/3* height or median_center_y < 1/3* height: 
        raise Exception("Error: Center dot outside of ROI (middle 1/3)")
    df_frame['d_center_dots'] = np.nan
    df_frame['d_fov_center'] = np.nan
    df_frame['flag_center_dot_outlier'] = 0
    df_frame['flag_fov_dot_outlier']= 0
    for i in range(len(df_frame)):
        # determine if center dot is outlier based on distance to the median locatioin, if >50px, mark as outlier
        df_frame['d_center_dots'].iloc[i] = np.sqrt((df_frame['center_dot_x'].iloc[i] - median_center_x)**2 + (df_frame['center_dot_y'].iloc[i] - median_center_y)**2)
        if df_frame['d_center_dots'].iloc[i] >50:
            df_frame['flag_center_dot_outlier'].iloc[i] = 1
            print ('Warning: center dot outlier detected on frame# ', str(df_frame['frame_num'].iloc[i]))

        # determine if FOV dot is outlier, if d<25px, mark as outlier, if y distance >200px, outlier   
        df_frame['d_fov_center'].iloc[i] = np.sqrt((df_frame['fov_dot_x'].iloc[i] - df_frame['center_dot_x'].iloc[i])**2 + (df_frame['fov_dot_y'].iloc[i] - df_frame['center_dot_y'].iloc[i])**2)        
        if df_frame['d_fov_center'].iloc[i] <25 or np.abs(df_frame['fov_dot_y'].iloc[i] - df_frame['center_dot_y'].iloc[i]) > 200:
            df_frame['flag_fov_dot_outlier'].iloc[i] = 1
            print ('Warning: FOV dot outlier detected on frame# ', str(df_frame['frame_num'].iloc[i]))
  
    xx, yy = np.meshgrid(np.linspace(-60, 60, 121), np.linspace(-60, 60, 121))
    map_fov = np.sqrt(xx**2 + yy**2)
    
    #find middle frame by min(d_fov_center)
    min_d_fov_center = np.min(df_frame['d_fov_center'][df_frame['flag_center_dot_outlier']==0][df_frame['flag_fov_dot_outlier']==0])
    center_frame_index = df_frame['index'][df_frame['d_fov_center'] == min_d_fov_center].tolist()[0]
    summary= df_frame[df_frame['index']==center_frame_index].to_dict(orient='records')[0]  #generate summary dict starting w center frame info 
    
    map_dxdy_median = calc_median_map(maps_dxdy, min_num =1)
    unitspacing_xy = map_dxdy_median[60,60]/dxdy_spacing
    
    [xi_fov,yi_fov] = [summary['fov_dot_x'] - summary['center_dot_x'],summary['fov_dot_y'] - summary['center_dot_y']]/unitspacing_xy
    summary['xi_fov'] = xi_fov
    summary['yi_fov'] = yi_fov
    print ('Middle frame found: #', str(summary['frame_num']),'; FOV dot @index ', [xi_fov,yi_fov])
    
    #Normalize map_dxdy for local PS 
    map_dxdy_norm = maps_dxdy[center_frame_index] / map_dxdy_median
    map_dxdy_norm_fov = offset_map_fov(map_dxdy_norm, xi_fov, yi_fov)
    summary_local = calc_parametrics_local (map_dxdy_norm_fov, map_fov)
    summary.update(summary_local)
    try: 
        plot_map_norm(map_dxdy_norm_fov)
    except ValueError:
        print('Error: plot local PS map')
    
    #Global PS
    map_distance = calc_distance(maps_xy[center_frame_index], maps_xy[center_frame_index][60,60,:])
    map_unit = map_distance / map_fov

    
    for i in [0,-1]: #first and last frame
        if i ==0:
            label = 'gazeright'
        else:
            label = 'gazeleft'
        frame_num = frame_nums[i]
        map_delta_global = maps_xy[i] - maps_xy[center_frame_index]
        map_distance_global = calc_distance(map_delta_global, map_delta_global[60,60,:])
        map_global = map_distance_global  / map_unit
        map_global = offset_map_fov(map_global, xi_fov, yi_fov)
        summary_global = calc_parametrics_global (map_global, map_fov, label, frame_num)
        summary.update(summary_global)
        try:
            plot_map_global(map_global, fname_str= label)
        except ValueError:
            print('Error: plot globle PS map')
    
    
    df.to_csv(csv_file)
    df_frame.to_csv(csv_file_frame)
    with open(csv_file_summary, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, summary.keys())
        w.writeheader()
        w.writerow(summary)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    #process_csv(csv_file)
    

        




