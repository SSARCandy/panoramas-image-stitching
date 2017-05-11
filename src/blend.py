# coding: utf-8

import numpy as np


# find best shift using RANSAC
def RANSAC(matched_pairs):
    matched_pairs = np.asarray(matched_pairs)
    
    best_shift = []
    K = 1000
    threshold_distance = 20
    
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


# img should bigger than img2
def blending(img1, img2, shift, pool):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, shift[1]),
        (0, 0)
    ]
    shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)
    splited = shifted_img1[:, img2.shape[1]:] if shift[1] > 0 else shifted_img1[:, :-img2.shape[1]]
    shifted_img1 = shifted_img1[:, :img2.shape[1]] if shift[1] > 0 else shifted_img1[:, -img2.shape[1]:]
    
    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape
    
    inv_shift = [h1-h2, w1-w2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shifted_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)


    shifted_img1 = pool.starmap(get_new_row_colors, [(shifted_img1[y], shifted_img2[y]) for y in range(h1)])
      
    shifted_img1 = np.asarray(shifted_img1)
    shifted_img1 = np.concatenate((shifted_img1, splited), axis=1)
    return shifted_img1

def get_new_row_colors(row1, row2):
    new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

    for x in range(len(row1)):
        color1 = row1[x]
        color2 = row2[x]
        if list(color1) == [0, 0, 0]:
            new_row[x] = color2
        elif list(color2) == [0, 0, 0]:
            new_row[x] = color1
        else:
            ratio = x/len(row1)
            if ((color1 - color2)**2).sum() > 10000:
                ratio = 1
            new_row[x] = (1-ratio)*color1 + ratio*color2

    return new_row


def blend_all(imgs, shifts):
    shifts = np.asarray(shifts)
    global_shift = np.sum(shifts, axis=0)

