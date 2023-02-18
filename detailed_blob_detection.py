#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Mon Nov 28 17:27:06 2022

@author: melshaer0612
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
        def calc_slopes(x, y, slope, error, ver=False):    
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
            if ver:
                dots_selected_y = np.asarray([dot.x for dot in dots_selected])
                dots_selected_x = np.asarray([dot.y for dot in dots_selected])
            else:
                dots_selected_x = np.asarray([-dot.x for dot in dots_selected])
                dots_selected_y = np.asarray([-dot.y for dot in dots_selected])    
            if len(dots_selected) > 1:
                (_slope, _) = np.polyfit(dots_selected_y, dots_selected_x, 1)
            else:
                _slope = slope
             
            return _slope
        
        x, y = self.get_x_y()
        self.hor_slope = calc_slopes(-y, -x, init_hor_slope, hor_dist_error) 
        
        x, y = self.get_x_y(inv=True)
        self.ver_slope = calc_slopes(y, x, init_ver_slope, ver_dist_error, ver=True)
        
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
            list_dots_left = np.vstack((x, y)).T
            list_dots_left = list_dots_left[y.argsort()]
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
                        dots_selected = np.fliplr(dots_selected)
                        list_lines.append(dots_selected)
                    else:
                        list_lines.append(dots_selected)
            list_len = [len(i) for i in list_lines]
            len_accepted = np.int16(accepted_ratio * np.max(list_len))
            lines_selected = [line for line in list_lines if len(line) > len_accepted] 
            if ver:
                lines_selected = sorted(lines_selected, key=lambda list_: np.mean(list_[:, 1]))
            else: 
                lines_selected = sorted(lines_selected, key=lambda list_: np.mean(list_[:, 0]))
            
            return lines_selected
        
        x, y = self.get_x_y()
        hor_lines = get_lines(x, y, self.hor_slope)
        self.hor_lines = [list(map(tuple, line)) for line in hor_lines]
        
        x, y = self.get_x_y(inv=True)
        ver_lines = get_lines(x, y, self.ver_slope, ver=True)
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
            print('Center Dot was not indexed. Will use the closest dot instead.')
            dist_2 = np.sum((np.asarray(common_pts) - np.asarray(center_dot_pt)) ** 2, axis=1)
            pseudo_center_pt = common_pts[np.argmin(dist_2)]
            center_dot_hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pseudo_center_pt in line])
            center_dot_vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pseudo_center_pt in line])
        
        for pt in common_pts:
            hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pt in line])
            vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pt in line])
            self.dotsxy_indexed[pt] = list(zip(vi - center_dot_vi, hi - center_dot_hi))
            
            
    def plot_lines_dots(self, width, height, filename):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width / 100, height / 100)
        for line in self.hor_lines:
            line = np.asarray(line)
            plt.plot(line[:, 0], height - line[:, 1], '-o', color='blue')
        for line in self.ver_lines:
            line = np.asarray(line)
            plt.plot(line[:, 0], height - line[:, 1], '-o', color='green')
        #plt.scatter(self.center_dot.x, self.center_dot.y)  
        plt.savefig(os.path.join(output_path, filename), dpi=100)
        #plt.show()
        

                    
class Dot:            
    def __init__(self, x, y, size, circularity=None, convexity=None, inertia=None):
        self.x = x
        self.y = y
        self.size = size
        self.circularity = circularity
        self.convexity = convexity
        self.inertia = inertia
            
    def __location__(self):
        return '({0}, {1})'.format(self.x, self.y)
    
    def __circularity__(self):
        return 'Circularity is {0}'.format(self.circularity)
    
    def __convexity__(self):
        return 'Convexity is {0}'.format(self.convexity)
    
    def __inertia__(self):
        return 'Inertia is {0}'.format(self.inertia)
    
    
current_path = os.getcwd()

# input_path = os.path.join(current_path, '60deg_test_55p5' + '/')
# output_path = os.path.join(current_path, '60deg_test_55p5_output' + '/')
# image_files = glob.glob(input_path + '*.tiff')

image_files = ['test4.tiff']
input_path = current_path
output_path = current_path

#----Xuan test------

# current_path = r'/Users/xuawang/Dropbox (Meta)/Pupil Swim Metrology/PST/20221121 2d dots 60deg/Arcata EVT2 A/test_55p5/'

# input_path = current_path
# output_path = os.path.join(current_path, 'output' + '/')
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# image_files = glob.glob(input_path + '*.tiff')
# image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
# image_files =image_files[:1]
# #image_files =image_files[:110:10]  #every nth image

#-------------

log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.txt'
csv_file = os.path.join(output_path, time.strftime('%Y%m%d-%H%M%S') + '_indexed_dots.csv')


# dot_params = {'min_dist' : 3,
#               'min_threshold' : 9, 
#               'max_threshold' : 240,
#               'threshold_step' : 3, 
#               'min_area' : 5, 
#               'max_area' : 2000, 
#               'invert' : True, 
#               'filter_byconvexity' : False,
#               'min_convexity': 0.6, 
#               'filter_bycircularity' : True}

dot_params = {'min_dist' : 3,
              'min_threshold' : 10,
              'max_threshold' : 50,
              'threshold_step' : 10, 
              'min_repeatability' : 2,
              'filter_byarea' : True,
              'min_area' : 5, 
              'max_area' : 2000, 
              'filter_byconvexity' : False, 
              'min_convexity' : 0.75, 
              'max_convexity' : np.inf,
              'filter_byinertia' : False,
              'filter_bycircularity' : False,
              'min_circularity' : 0.8,
              'max_circularity' : np.inf,
              'min_inertia_ratio' : 0.1,
              'max_inertia_ratio' : np.inf}


def write_log(*args, filename):
    log_file = open(os.path.join(output_path, filename), 'a+')
    line = ' '.join([str(a) for a in args])
    log_file.write(line + '\n')
    print(line)
    

def distance_kps(dot1, dot2):
    return np.sqrt((dot1.x - dot2.x) ** 2 + (dot1.y - dot2.y) ** 2)


def detect_blobs(gray, binarized):
    blobs = []
    contours, _ = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #------------------
    # For debugging only
    # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # image_cpy = image.copy()
    # cv2.drawContours(image_cpy, contours, -1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.imshow('Contours', image_cpy)
    # cv2.waitKey(0)
    #------------------
    for contour in contours:
        circ = None 
        cnvx = None
        inert = None
        moms = cv2.moments(contour)
        if dot_params['filter_byarea']:
            area = moms['m00']
            
            if area < dot_params['min_area'] or area >= dot_params['max_area']:
                continue
            
        if dot_params['filter_bycircularity']:
            area = moms['m00']
            perimeter = cv2.arcLength(contour, True)
            circ = 4 * np.pi * area / (perimeter ** 2)
            
            if circ < dot_params['min_circularity'] or circ >= dot_params['max_circularity']:
                continue
            
        if dot_params['filter_byinertia']:
            denominator = np.sqrt((2 * moms['mu11']) ** 2 + (moms['mu20'] - moms['mu02']) ** 2)
            if denominator > 1e-2:
                cosmin = (moms['mu20'] - moms['mu02']) / denominator
                sinmin = 2 * moms['mu11'] / denominator
                cosmax = - cosmin
                sinmax = - sinmin
                
                imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] 
                                                                    - moms['mu02']) * cosmin - moms['mu11'] * sinmin
                imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] 
                                                                    - moms['mu02']) * cosmax - moms['mu11'] * sinmax
                inert = imin / imax
            else:
                inert = 1
            
            if inert < dot_params['min_inertia_ratio'] or inert >= dot_params['max_inertia_ratio']:
                continue 
            
        if dot_params['filter_byconvexity']:
            hull = cv2.convexHull(contour)
            area = moms['m00']
            hull_area = cv2.contourArea(hull)
            if np.abs(hull_area) < np.finfo(float).eps:
                continue
            cnvx = area / hull_area
            
            if cnvx < dot_params['min_convexity'] or cnvx >= dot_params['max_convexity']:
                continue
            
        if moms['m00'] == 0.0:
            continue
        
        # Find dot location
        x = moms['m10'] / moms['m00']
        y = moms['m01'] / moms['m00']
        
        # Find dot size
        dist_list = []
        for pt in contour:
            x_pt = pt[0][0]
            y_pt = pt[0][1]
            dist_list.append(np.sqrt((x_pt - x) ** 2 + (y_pt - y) ** 2)) 
            
        dist_list = sorted(dist_list)
        size = (dist_list[int((len(dist_list) - 1) / 2)] + dist_list[int(len(dist_list) / 2)]) / 2
        
        #Find dot circularity
        if circ is None:
            perimeter = cv2.arcLength(contour, True)
            circ = 4 * np.pi * moms['m00'] / (perimeter ** 2)
            
        #Find dot convexity
        if cnvx is None:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if np.abs(hull_area) < np.finfo(float).eps:
                continue
            cnvx = moms['m00'] / hull_area
        
        #Find dot inertia
        if inert is None:
            denominator = np.sqrt((2 * moms['mu11']) ** 2 + (moms['mu20'] - moms['mu02']) ** 2)
            if denominator > 1e-2:
                cosmin = (moms['mu20'] - moms['mu02']) / denominator
                sinmin = 2 * moms['mu11'] / denominator
                cosmax = - cosmin
                sinmax = - sinmin
                
                imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] 
                                                                    - moms['mu02']) * cosmin - moms['mu11'] * sinmin
                imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] 
                                                                    - moms['mu02']) * cosmax - moms['mu11'] * sinmax
                inert = imin / imax
            else:
                inert = 1
        
        blobs.append(Dot(x, y, size, circ, cnvx, inert))
        
    # image_cpy = image.copy()    
    # for dot in blobs:
    #     cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size/2), (0, 0, 255), 3)
    # cv2.imshow('Blobs', image_cpy)
    # cv2.waitKey(0)
    
    return blobs
    
        
def find_dots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert(dot_params['min_threshold'] + dot_params['threshold_step'] <= dot_params['max_threshold'])
    thresh_list = np.arange(start=dot_params['min_threshold'], stop=dot_params['max_threshold'], step=dot_params['threshold_step'])
    blobs = []
    for thresh in thresh_list:
        print('Threshold:', thresh)
        _, binarized = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        cur_blobs = detect_blobs(gray, binarized)
        new_blobs = []
        for cur_blob_idx in range(len(cur_blobs)):
            is_new = True
            cur_dot = cur_blobs[cur_blob_idx]
            if len(blobs) != 0:
                for blob_idx in range(len(blobs)):
                    dot = blobs[blob_idx][int(len(blobs[blob_idx]) / 2)]
                    dist = np.sqrt((dot.x - cur_dot.x) ** 2 + (dot.y - cur_dot.y) ** 2)
                    is_new = (dist >= dot_params['min_dist']) and (dist >= dot.size) and (dist >= cur_dot.size)
                    if not is_new:
                        blobs[blob_idx].append(cur_dot)
                        k = len(blobs[blob_idx]) - 1
                        while k > 0 and cur_dot.size < blobs[blob_idx][k-1].size:
                            blobs[blob_idx][k] = blobs[blob_idx][k-1]
                            k = k - 1
                        blobs[blob_idx][k] = cur_dot
                        break                
            if is_new:
                new_blobs.append([cur_dot])
        blobs.extend(new_blobs)
        
    frame = Frame()    
    for i in range(len(blobs)):
        if len(blobs[i]) < dot_params['min_repeatability']:
            continue
        x = 0
        y = 0
        norm = 0
        for j in range(len(blobs[i])):
            dot = blobs[i][j]
            conf = dot.inertia ** 2
            x = x + conf * dot.x
            y = y + conf * dot.y
            norm = norm + conf
        x = x * (1 / norm)
        y = y * (1 / norm)
        size = blobs[i][int(len(blobs[i]) / 2)].size * 2
        circ = blobs[i][int(len(blobs[i]) / 2)].circularity
        cnvx = blobs[i][int(len(blobs[i]) / 2)].convexity
        inert = blobs[i][int(len(blobs[i]) / 2)].inertia
        frame.dots.append(Dot(x, y, size, circ, cnvx, inert))
    
    return frame
    

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
    cv2.imwrite(os.path.join(output_path, filename), image_cpy)
    

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
    
    
if __name__ == '__main__':
    start_time = time.monotonic()
    df = pd.DataFrame()
    cnt = 0
    for image_file in image_files:
        frame_num = ((image_file.split(os.path.sep)[-1].split('_'))[-1].split('.tiff'))[0]
        image = cv2.imread(os.path.join(current_path, image_file))
        print('Frame', frame_num, ': Processing started')
        height, width, _ = image.shape
        frame = find_dots(image)
        print('Frame', frame_num, ': Finding dots is complete, found ', str(len(frame.dots)), 'dots')

        




