#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:27:06 2022

@author: melshaer0612

# Version Complete -- 01/25/2023

"""

import os
import cv2
import numpy as np
import config.config as cf
import pst_algo.__main__ as mn

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
        dots_array = np.vstack((x, y)).T
        dist = [np.sort(np.sqrt((dot[0] - dots_array[:, 0]) ** 2 + (dot[1] - dots_array[:, 1]) ** 2))[1]
                      for dot in dots_array] 
        
        self.med_dot_dist = np.median(dist)
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
        mapdx = mn.shift_fillnan(mapx,-pos,0) - mn.shift_fillnan(mapx,-neg,0)
        mapy = map_xy_copy[:,:,1]
        mapdy = mn.shift_fillnan(mapy,-pos,1) - mn.shift_fillnan(mapy,-neg,1)
        
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
       cv2.imwrite(os.path.join(cf.output_path, filename), image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])
        

                    
class Dot:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
            
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
    

def distance_kps(dot1, dot2):
    return np.sqrt((dot1.x - dot2.x) ** 2 + (dot1.y - dot2.y) ** 2)


def detect_blobs(image):
    if cf.params['invert']:
        image = 255 - image

    if cf.params['subtract_median']:
        med_blur = cv2.medianBlur(image, 11)
        image = 255 - 255 / (np.max(np.abs(med_blur - image))) * np.abs(med_blur - image)
        
    # Setup SimpleBlobDetector parameters.
    blob_params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    blob_params.minThreshold = int(cf.params['min_threshold'])
    blob_params.maxThreshold = int(cf.params['max_threshold'])
    blob_params.thresholdStep = int(cf.params['threshold_step'])
    blob_params.minDistBetweenBlobs = int(cf.params['min_dist'])
    # Filter by Area
    blob_params.filterByArea = True
    blob_params.minArea = int(cf.params['min_area'])
    blob_params.maxArea = int(cf.params['max_area'])
    # Filter by Circularity
    blob_params.filterByCircularity = cf.params['filter_bycircularity']
    blob_params.filterByInertia = False
    blob_params.filterByConvexity = cf.params['filter_byconvexity']
    blob_params.minConvexity = cf.params['min_convexity']   # 0.9 filters out most noise at center, lost a few at edge
    
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
    x, y, size = detect_blobs(image)
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
        return Dot(0, 0, 0)
    
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
        _, thresh = cv2.threshold(gray, cf.params['bin_th'], 255, cv2.THRESH_BINARY)
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
    cv2.imwrite(os.path.join(cf.output_path, filename), image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])
