# coding: utf-8

import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import feature
import utils
import blend

img_list = utils.load_images('../input_image/parrington')
cylinder_img_list = [utils.cylindrical_projection(img, 706) for img in img_list]

blended_image = cylinder_img_list[0].copy()
for img in cylinder_img_list[1:]:
    print('Find corner response 1')
    corner_response1 = feature.harris_corner(blended_image)
    descriptors1, position1 = feature.extract_description(blended_image, corner_response1, kernel=5)

    print('Find corner response 2')
    corner_response2 = feature.harris_corner(img)
    descriptors2, position2 = feature.extract_description(img, corner_response2, kernel=5)
    
    print('Feature matching')
    mp = feature.matching(descriptors1, descriptors2, position1, position2)

    print('Find best shift using RANSAC')
    shift = blend.RANSAC(mp)

    print('Blending image')
    blended_image = blend.blending(blended_image, img, shift)
    
    cv2.imshow('g', blended_image)
    cv2.waitKey(0)