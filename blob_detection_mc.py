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

import time
from datetime import timedelta

from skimage.transform import radon


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
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
            
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
        
    
current_path = os.getcwd()

input_path = os.path.join(current_path, '60deg_test_55p5' + '/')
output_path = os.path.join(current_path, '60deg_test_55p5_output' + '/')
image_files = glob.glob(input_path + '*.tiff')

# image_files = ['test2.tiff']
# input_path = current_path
# output_path = current_path

log_file = 'Detection Log_' + time.strftime('%Y%m%d-%H%M%S') + '.txt'


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
    cv2.imwrite(os.path.join(output_path, filename), image_cpy)
    
    
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
        print('Frame', frame_num, ': Finding dots is complete')
        
        frame.center_dot = find_center_dot(frame.dots, height, width)
        print('Frame', frame_num, ': Center dot was found at ', frame.center_dot.__str__())
        
        fov_dot = find_fov(image, height, width)
        print('Frame', frame_num, ': Finding fov dot is complete')
        if fov_dot is None:
            write_log('ERROR: Could not find FOV point for frame ', str(frame_num), filename=log_file)          
        else:
            draw_dots(image, [fov_dot], filename=frame_num+'_fov.tiff') # For debugging FOV dot detection
        
        draw_dots(image, frame.dots, filename=frame_num+'_dots.tiff') # For debugging blob detection
        
        med_size, med_dist = frame.calc_dot_size_dist()
        print('Dot Size:', med_size, 'Distance:', med_dist)
        
        print('Starting slope calculations for frame', frame_num)
        proc = prep_image(image, normalize_and_filter=True, binarize=False)
        init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = get_initial_slopes(proc, height, width, ratio=0.3)
        hor_slope, ver_slope = frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
        print('HSlope:', hor_slope, 'VSlope:', ver_slope)
        hor_lines, ver_lines = frame.group_lines()
        frame.plot_lines_dots(width, height, filename=frame_num+'_grouped.tiff')
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
        print('Total frames processed is', str(cnt), '/ 150')
        
    df.to_csv(os.path.join(output_path,'indexed_dots.csv'))
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    

        




