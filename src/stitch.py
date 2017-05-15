# coding: utf-8

import numpy as np
import cv2
import feature

# find best shift using RANSAC
def RANSAC(matched_pairs):
    matched_pairs = np.asarray(matched_pairs)
    
    best_shift = []
    K = 1000
    threshold_distance = 3
    
    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        random_idx = int(np.random.random_sample()*len(matched_pairs))
        sample = matched_pairs[random_idx]
        
        # fit the warp model
        shift = sample[1] - sample[0]
        
        # calculate inliner points
        shifted = matched_pairs[:,1] - shift
        difference = matched_pairs[:,0] - shifted
        
        inliner = 0
        for diff in difference:
            if (diff**2).sum()**0.5 < threshold_distance:
                inliner = inliner + 1
        
        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = shift
        
    return best_shift



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
        shifted_img1 = pool.starmap(alpha_blend, [(shifted_img1[y], shifted_img2[y], seam_x, 2, direction) for y in range(h1)])
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


def end2end_align(img, pool):
    p1 = img[:,:500]
    p2 = img[:,-500:]

    cr1 = feature.harris_corner(p1, pool)
    ds1, pos1 = feature.extract_description(p1, cr1, kernel=5, threshold=0.05)

    cr2 = feature.harris_corner(p2, pool)
    ds2, pos2 = feature.extract_description(p2, cr2, kernel=5, threshold=0.05)

    mp =  feature.matching(ds1, ds2, pos1, pos2, pool, y_range=float('inf'))

    y_shift, _ = RANSAC(mp)

    aligned = img.copy()
    col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
    for x in range(img.shape[1]):
        aligned[:,x] = np.roll(img[:,x], col_shift[x], axis=0)

    return aligned

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
