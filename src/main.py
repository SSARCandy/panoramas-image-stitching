# coding: utf-8

import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import feature
import utils
import stitch


DEBUG=False


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[Usage] python script <input img dir>')
        print('[Exampe] python script ../input_image/parrington')
        sys.exit(0)
    
    input_dirname = sys.argv[1]
    

    pool = mp.Pool(mp.cpu_count())

    img_list, focal_length = utils.parse(input_dirname)
    
    # img_list = img_list[:3]

    print('Warp images to cylinder')
    cylinder_img_list = pool.starmap(utils.cylindrical_projection, [(img_list[i], focal_length[i]) for i in range(len(img_list))])


    _, img_width, _ = img_list[0].shape
    stitched_image = cylinder_img_list[0].copy()

    shifts = [[0, 0]]

    # add first img for end to end align
    #cylinder_img_list += [stitched_image]
    for i in range(1, len(cylinder_img_list)):
        print('Computing .... '+str(i+1)+'/'+str(len(cylinder_img_list)))
        img1 = cylinder_img_list[i-1]
        img2 = cylinder_img_list[i]

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
        matched_pairs = feature.matching(descriptors1, descriptors2, position1, position2, pool, y_range=10)
        print(str(len(matched_pairs)) +' features matched.')

        if DEBUG:
            utils.matched_pairs_plot(img1, img2, matched_pairs)

        print(' - Find best shift using RANSAC .... ', end='', flush=True)
        shift = stitch.RANSAC(matched_pairs)
        shifts += [shift]
        print('best shift ', shift)

        print(' - Stitching image .... ', end='', flush=True)
        stitched_image = stitch.stitching(stitched_image, img2, shift, pool, blending=True)
        cv2.imwrite(str(i) +'.jpg', stitched_image)
        print('Saved.')


    print('Perform end to end alignment')
    aligned = stitch.end2end_align(stitched_image, pool)
    cv2.imwrite('aligned.jpg', aligned)

    print('Cropping image')
    cropped = stitch.crop(aligned)
    cv2.imwrite('cropped.jpg', cropped)