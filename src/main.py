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

print('Warp images to cylinder')
cylinder_img_list = [utils.cylindrical_projection(img, 706) for img in img_list]

blended_image = cylinder_img_list[0].copy()
for i in range(1, len(cylinder_img_list)):
    img1 = cylinder_img_list[i-1]
    img2 = cylinder_img_list[i]

    print('Find corner response 1')
    corner_response1 = feature.harris_corner(img1)
    descriptors1, position1 = feature.extract_description(img1, corner_response1, kernel=5)

    print('Find corner response 2')
    corner_response2 = feature.harris_corner(img2)
    descriptors2, position2 = feature.extract_description(img2, corner_response2, kernel=5)
    
    print('Feature matching')
    mp = feature.matching(descriptors1, descriptors2, position1, position2)
    print('Find '+ str(len(mp)) +' matched features')

    print('Find best shift using RANSAC')
    shift = blend.RANSAC(mp)

    print('Blending image')
    blended_image = blend.blending(blended_image, img2, shift)
    
    cv2.imwrite('tmp'+ str(i) +'.jpg', blended_image)