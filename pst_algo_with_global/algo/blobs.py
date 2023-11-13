#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: melshaer0612@meta.com

"""

import sys
import cv2
import numpy as np

from skimage.transform import radon
from algo.structs import Frame, Dot


sys.dont_write_bytecode = True # Disables __pycache__


def dot_pattern(image, height, width, params, frame_num, logger):
    frame = find_dots(image, params)
    logger.info('Frame %s : Finding dots is complete, found %d dots', frame_num, len(frame.dots))
        
    frame.center_dot = find_center_dot(frame.dots, height, width)
    logger.info('Frame %s : Center dot was found at %s', frame_num, frame.center_dot.__str__())

    frame.calc_dot_size_dist()
    logger.info('Frame %s Dot Size: %0.2f Distance: %0.2f', frame_num, frame.med_dot_size, frame.med_dot_dist)
        
    logger.info('Starting slope calculations for frame %s', frame_num)
    proc = prep_image(image, params, normalize_and_filter=True, binarize=False)
    init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error = get_initial_slopes(proc, height, width, ratio=0.3)
    frame.get_slopes(init_hor_slope, init_ver_slope, hor_dist_error, ver_dist_error)
    logger.info('Frame %s HSlope: %0.2f VSlope: %0.2f', frame_num, frame.hor_slope, frame.ver_slope)
        
    frame.group_lines()
    frame.find_index(logger, frame_num)
    logger.info('Finished indexing calculations for frame %s', frame_num)
        
    frame.generate_map_xy(logger, frame_num)
    frame.generate_map_dxdy(params['dxdy_spacing'])  #4x spacing

    return frame


def distance_kps(dot1, dot2):
    return np.sqrt((dot1.x - dot2.x) ** 2 + (dot1.y - dot2.y) ** 2)


def detect_blobs(image, params):
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
    blob_params.filterByCircularity = params['filter_bycircularity']
    blob_params.filterByInertia = False
    blob_params.filterByConvexity = params['filter_byconvexity']
    blob_params.minConvexity = params['min_convexity']   # 0.9 filters out most noise at center, lost a few at edge
    
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(image.astype('uint8'))
    x = [kp.pt[0] for kp in keypoints]
    y = [kp.pt[1] for kp in keypoints]
    size = [kp.size for kp in keypoints] # Diameter of blob

    return x, y, size


def find_center_dot(dots, height, width):
    # Find center dot by size, within the ROI (this is to avoid large blob at other areas)
    max_size = 0
    max_size_index = 0
    center_roi = 1 / 6 # find the center spot within the center of the image only
    for kp in range(len(dots)):
        if dots[kp].x > (0.5 - 0.5 * center_roi) * width and dots[kp].x < (0.5 + 0.5 * center_roi) * width:  # x filter
            if dots[kp].y > (0.5 - 0.5 * center_roi) * height and dots[kp].y < (0.5 + 0.5 * center_roi) * height: # y filter
                if dots[kp].size > max_size:
                    max_size = dots[kp].size
                    max_size_index = kp
    center_dot = Dot(dots[max_size_index].x, dots[max_size_index].y, dots[max_size_index].size) # creates a new object
    
    return center_dot
    
        
def find_dots(image, params):
    image = prep_image(image, params, normalize_and_filter=True, binarize=False)
    x, y, size = detect_blobs(image, params)
    frame = Frame()
    frame.dots = [Dot(x_i, y_i, size_i) for (x_i, y_i, size_i) in list(zip(x, y, size))]
    
    return frame  
  
    
def find_fov(image, params, logger, frame_num, height, width):
    max_fov_dot_size = 30
    roi_hei = np.int16(height / 3)
    roi_image = image[roi_hei:height - roi_hei]
    image = prep_image(roi_image, params, normalize_and_filter=True, binarize=True)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    outer_c = np.where((hierarchy[0,:,2] != -1) & (hierarchy[0,:,3] == -1))[0].tolist()
    inner_c = np.where((hierarchy[0,:,2] == -1) & (hierarchy[0,:,3] != -1))[0].tolist()
    cand_fov_dots = []

    #-- Amending this -----------------------------------------
    # size_lst = []
    # if len(outer_c) > 0 and len(inner_c) > 0:
    #     cands = list(zip(outer_c, inner_c))
    #     for cand in cands:
    #         M_inner = cv2.moments(contours[cand[1]])
    #         M_outer = cv2.moments(contours[cand[0]])
    #         try:
    #             x = M_inner['m10'] / M_inner['m00']
    #             y = M_inner['m01'] / M_inner['m00'] + roi_hei
    #             size = M_outer['m00'] #size = M_outer['m00'] - M_inner['m00']
    #             cand_fov_dots.append(Dot(x, y, size))
    #             size_lst.append(size)
    #         except ZeroDivisionError:
    #             continue
    #     if len(size_lst) > 0:              
    #         fov_dot = [dot for dot in cand_fov_dots if dot.size == np.max(size_lst)][0]
    #     else:
    #         fov_dot = Dot(width / 2, height / 2, 0)
    #         # fov_dot = Dot(0, 0, 0)
    #         logger.error('Frame %s: Error finding FOV dot', frame_num)
    # else:
    #     fov_dot = Dot(width / 2, height / 2, 0)
    #     # fov_dot = Dot(0, 0, 0)
    #     logger.error('Frame %s: Error finding FOV dot', frame_num) 

    # --With this--------------------------------------------------------

    # Take a list of outer contours and match them with their inner contours 
    cands = []
    for parent in outer_c:
        if len(np.where(hierarchy[0,:,3] == parent)[0]) == 1:
            cands.append([parent, np.where(hierarchy[0,:,3] == parent)[0][0]])
        
    if len(outer_c) > 0 and len(inner_c) > 0:
        for cand in cands:
            M_inner = cv2.moments(contours[cand[1]])
            M_outer = cv2.moments(contours[cand[0]])
            try:
                x = M_inner['m10'] / M_inner['m00']
                y = M_inner['m01'] / M_inner['m00'] + roi_hei
                size = np.sqrt(M_outer['m00'] / np.pi) * 2
                cand_fov_dots.append(Dot(x, y, size))
            except ZeroDivisionError:
                continue
        if len(cand_fov_dots) > 0:
            fov_dot = cand_fov_dots[np.where(np.array([dot.size for dot in cand_fov_dots]) < max_fov_dot_size)[0][0]]
        else:
            fov_dot = Dot(width / 2, height / 2, 0)
            # fov_dot = Dot(0, 0, 0)
            logger.error('Frame %s: Error finding FOV dot', frame_num)
    else:
        fov_dot = Dot(width / 2, height / 2, 0)
        # fov_dot = Dot(0, 0, 0)
        logger.error('Frame %s: Error finding FOV dot', frame_num)
        
    return fov_dot


def prep_image(image, params, normalize_and_filter, binarize):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if normalize_and_filter:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 3, 100, 100)
    if binarize:
        _, thresh = cv2.threshold(gray, params['binary_threshold'], 255, cv2.THRESH_BINARY)
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
    coarse_range = 10.0  # Degree
    
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


def draw_dots(image, dots, filepath, enable=True):
    if enable:
        image_cpy = image.copy()
        for idx, dot in enumerate(dots):
            if len(dots) > 2:
                cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size / 2), (0, 255, 0), 3) # Green Dots
            elif len(dots) == 2 and idx == 0:
                cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size / 2), (0, 0, 255), 3) # Red Circle+ 
            elif len(dots) == 2 and idx == 1:
                cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size / 2), (0, 255, 0), 3) # Center Dot+
            elif (len(dots) == 1) and ('fov' in filepath):
                cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size / 2), (0, 0, 255), 3) # Just Red Circle
            elif (len(dots) == 1) and ('center' in filepath):
                cv2.circle(image_cpy, (int(dot.x), int(dot.y)), int(dot.size / 2), (0, 255, 0), 3) # Just Center Dot
        cv2.imwrite(filepath, image_cpy, [cv2.IMWRITE_JPEG_QUALITY, 40])
