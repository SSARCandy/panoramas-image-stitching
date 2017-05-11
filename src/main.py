# coding: utf-8

import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import feature
import utils
import blend



if __name__ == '__main__':
    DEBUG=False
    pool = mp.Pool(mp.cpu_count())

    img_list, focal_length = utils.parse('../input_image/parrington2')
    
    print('Warp images to cylinder')
    cylinder_img_list = pool.starmap(utils.cylindrical_projection, [(img_list[i], focal_length[i]) for i in range(len(img_list))])

    shifts = []
    direction = ''
    _, img_width, _ = img_list[0].shape
    blended_image = cylinder_img_list[0].copy()

    # add first img for end to end align
    cylinder_img_list += [blended_image]
    for i in range(1, len(cylinder_img_list)):
        print('Computing .... '+str(i+1)+'/'+str(len(cylinder_img_list)))
        img1 = blended_image
        img2 = cylinder_img_list[i]

        if direction != '':
            img1 = blended_image[:, :img_width] if direction == 'left' else blended_image[:, -img_width:]

        print(' - Find features in previous img .... ', end='', flush=True)
        corner_response1 = feature.harris_corner(img1, pool)
        descriptors1, position1 = feature.extract_description(img1, corner_response1, kernel=5, threshold=0.05)
        print(str(len(descriptors1))+' features extracted.')

        print(' - Find features in img_'+str(i+1)+' .... ', end='', flush=True)
        corner_response2 = feature.harris_corner(img2, pool)
        descriptors2, position2 = feature.extract_description(img2, corner_response2, kernel=5, threshold=0.05)
        print(str(len(descriptors2))+' features extracted.')

        if False:
            cv2.imshow('cr1', corner_response1)
            cv2.imshow('cr2', corner_response2)
            cv2.waitKey(0)
        
        print(' - Feature matching .... ', end='', flush=True)
        matched_pairs = feature.matching(descriptors1, descriptors2, position1, position2, y_range=10)
        print(str(len(matched_pairs)) +' features matched.')

        if DEBUG:
            utils.matched_pairs_plot(cylinder_img_list[i-1], img2, matched_pairs)

        print(' - Find best shift using RANSAC .... ', end='', flush=True)
        shift = blend.RANSAC(matched_pairs)
        shifts += [shift]
        print('best shift ', shift)
        if direction == '':
            direction = 'left' if shift[1] > 0 else 'right'

        print(' - Blending image .... ', end='', flush=True)
        blended_image = blend.blending(blended_image, img2, shift, pool)
        cv2.imwrite(''+ str(i) +'.jpg', blended_image)
        print('Saved.')

    print('Perform end to end alignment')
    aligned = blend.end2end_align(blended_image, pool)
    cv2.imwrite('aligned.jpg', aligned)

    print('Cropping image')
    cropped = blend.crop(aligned)
    cv2.imwrite('cropped.jpg', cropped)