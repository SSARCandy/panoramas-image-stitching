# coding: utf-8

import numpy as np
import cv2
import constant as const

"""
Find best shift using RANSAC

Args:
    matched_pairs: matched pairs of feature's positions, its an Nx2x2 matrix
    prev_shift: previous shift, for checking shift direction.

Returns:
    Best shift [y x]. ex. [4 234]

Raise:
    ValueError: Shift direction NOT same as previous shift.
"""
def RANSAC(matched_pairs, prev_shift):
    matched_pairs = np.asarray(matched_pairs)
    
    use_random = True if len(matched_pairs) > const.RANSAC_K else False

    best_shift = []
    K = const.RANSAC_K if use_random else len(matched_pairs)
    threshold_distance = const.RANSAC_THRES_DISTANCE
    
    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        idx = int(np.random.random_sample()*len(matched_pairs)) if use_random else k
        sample = matched_pairs[idx]
        
        # fit the warp model
        shift = sample[1] - sample[0]
        
        # calculate inliner points
        shifted = matched_pairs[:,1] - shift
        difference = matched_pairs[:,0] - shifted
        
        inliner = 0
        for diff in difference:
            if np.sqrt((diff**2).sum()) < threshold_distance:
                inliner = inliner + 1
        
        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = shift

    if prev_shift[1]*best_shift[1] < 0:
        print('\n\nBest shift:', best_shift)
        raise ValueError('Shift direction NOT same as previous shift.')

    return best_shift


"""
Stitch two image with blending.

Args:
    img1: first image
    img2: second image
    shift: the relative position between img1 and img2
    pool: for multiprocessing
    blending: using blending or not

Returns:
    A stitched image
"""
def stitching(img1, img2, shift, pool, blending=True):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

    # cut out unnecessary region
    split = img2.shape[1]+abs(shift[1])
    splited = shifted_img1[:, split:] if shift[1] > 0 else shifted_img1[:, :-split]
    shifted_img1 = shifted_img1[:, :split] if shift[1] > 0 else shifted_img1[:, -split:]

    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape
    
    inv_shift = [h1-h2, w1-w2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shifted_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)

    direction = 'left' if shift[1] > 0 else 'right'

    if blending:
        seam_x = shifted_img1.shape[1]//2
        tasks = [(shifted_img1[y], shifted_img2[y], seam_x, const.ALPHA_BLEND_WINDOW, direction) for y in range(h1)]
        shifted_img1 = pool.starmap(alpha_blend, tasks)
        shifted_img1 = np.asarray(shifted_img1)
        shifted_img1 = np.concatenate((shifted_img1, splited) if shift[1] > 0 else (splited, shifted_img1), axis=1)
    else:
        raise ValueError('I did not implement "blending=False" ^_^')

    return shifted_img1

def alpha_blend(row1, row2, seam_x, window, direction='left'):
    if direction == 'right':
        row1, row2 = row2, row1

    new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

    for x in range(len(row1)):
        color1 = row1[x]
        color2 = row2[x]
        if x < seam_x-window:
            new_row[x] = color2
        elif x > seam_x+window:
            new_row[x] = color1
        else:
            ratio = (x-seam_x+window)/(window*2)
            new_row[x] = (1-ratio)*color2 + ratio*color1

    return new_row

"""
End to end alignment

Args:
    img: panoramas image
    shifts: all shifts for each image in panoramas

Returns:
    A image that fixed the y-asix shift error
"""
def end2end_align(img, shifts):
    sum_y, sum_x = np.sum(shifts, axis=0)

    y_shift = np.abs(sum_y)
    col_shift = None

    # same sign
    if sum_x*sum_y > 0:
        col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
    else:
        col_shift = np.linspace(0, y_shift, num=img.shape[1], dtype=np.uint16)

    aligned = img.copy()
    for x in range(img.shape[1]):
        aligned[:,x] = np.roll(img[:,x], col_shift[x], axis=0)

    return aligned


"""
Crop the black border in image

Args:
    img: a panoramas image

Returns:
    Cropped image
"""
def crop(img):
    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upper, lower = [-1, -1]

    black_pixel_num_threshold = img.shape[1]//100

    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            upper = y
            break
        
    for y in range(thresh.shape[0]-1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            lower = y
            break

    return img[upper:lower, :]
