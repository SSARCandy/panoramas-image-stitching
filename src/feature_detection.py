# coding: utf-8

import cv2
import numpy as np


def harris_corner(img, k=0.04, block_size=2, kernel=11):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)/255

    corner_response = np.zeros(shape=gray.shape, dtype=np.float32)
    
    height, width, _ = img.shape
    dx = cv2.Sobel(gray, -1, 1, 0)
    dy = cv2.Sobel(gray, -1, 0, 1)
    Ixx = dx*dx
    Iyy = dy*dy
    Ixy = dx*dy
    
    cov_xx = cv2.boxFilter(Ixx, -1, (block_size, block_size), normalize=False)
    cov_yy = cv2.boxFilter(Iyy, -1, (block_size, block_size), normalize=False)
    cov_xy = cv2.boxFilter(Ixy, -1, (block_size, block_size), normalize=False)

    for y in range(height):
        for x in range(width):
            xx = cov_xx[y][x]
            yy = cov_yy[y][x]
            xy = cov_xy[y][x]

            det_M = xx*yy - xy**2
            trace_M = xx + yy
            
            R = det_M - k*trace_M**2
            corner_response[y][x] = R
            
    return corner_response
    

