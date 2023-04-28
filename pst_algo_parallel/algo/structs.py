#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import cv2
import numpy as np


class Dot:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
            
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
    

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
        self.xi_range = 60
        self.yi_range = 60
        self.map_xy = np.empty((self.xi_range * 2 + 1, self.yi_range * 2 + 1, 2)) # init map w +/-60 deg
        self.map_xy[:, :, :] = np.nan
        self.map_dxdy = np.empty((self.xi_range * 2 + 1, self.yi_range * 2 + 1, 2))
        self.map_dxdy[:, :, :] = np.nan
   
        
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
        self.hor_lines = [list(map(tuple, line)) for line in hor_lines]
        
        x, y = self.get_x_y()
        ver_lines = get_lines(x, y, self.ver_slope, ver=True)
        self.ver_lines = [list(map(tuple, line)) for line in ver_lines]
        
        return self.hor_lines, self.ver_lines
    
    
    def find_index(self, logger, frame_num):
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
            logger.warning('Frame %s: Center Dot was not indexed. Will use the closest dot instead.', frame_num)
            dist_2 = np.sum((np.asarray(common_pts) - np.asarray(center_dot_pt)) ** 2, axis=1)
            pseudo_center_pt = common_pts[np.argmin(dist_2)]
            center_dot_hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pseudo_center_pt in line])
            center_dot_vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pseudo_center_pt in line])
            return [],[]
        
        for pt in common_pts:
            hi = np.asarray([i_line for i_line, line in enumerate(self.hor_lines) if pt in line])
            vi = np.asarray([i_line for i_line, line in enumerate(self.ver_lines) if pt in line])
            self.dotsxy_indexed[pt] = list(zip(vi - center_dot_vi, hi - center_dot_hi))
            
            
    def generate_map_xy(self, logger, frame_num):
        # generate xy coordinate map from indexed dots
        cnt = 0
        for coord in self.dotsxy_indexed:
            ind = self.dotsxy_indexed[coord][0]
            if np.abs(ind[0]) <= self.xi_range:
                if np.abs(ind[1]) <= self.yi_range:
                    self.map_xy[ind[0] + self.xi_range, ind[1] + self.yi_range, :] = coord # offset center
                    cnt += 1
        logger.info('Frame %s: Map generation complete with %d indexed dots', frame_num, cnt)
        

    def generate_map_dxdy(self, spacing:int):
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
        
        # generate derivative map from map_xy, spacing as int
        pos = int(np.round(spacing/2))
        neg = pos - spacing
        map_xy_copy = np.copy(self.map_xy)
        mapx = map_xy_copy[:, :, 0]
        mapdx = shift_fillnan(mapx, -pos, 0) - shift_fillnan(mapx, -neg, 0)
        mapy = map_xy_copy[:, :, 1]
        mapdy = shift_fillnan(mapy, -pos, 1) - shift_fillnan(mapy, -neg, 1)
        
        self.map_dxdy[:, :, 0] = mapdx
        self.map_dxdy[:, :, 1] = mapdy
        
            
    def draw_lines_on_image(self, image, width, height, filepath):
       image_cpy = image.copy()
       for line in self.hor_lines:
           line_int = [(int(pt[0]), int(pt[1])) for pt in line]
           for pt1, pt2 in zip(line_int, line_int[1:]):
               cv2.line(image_cpy, pt1, pt2, [255, 0, 0], 2)
       for line in self.ver_lines:
           line_int = [(int(pt[0]), int(pt[1])) for pt in line]
           for pt1, pt2 in zip(line_int, line_int[1:]):
               cv2.line(image_cpy, pt1, pt2, [0, 255, 0], 2)
       cv2.imwrite(filepath, image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])
       
       
       