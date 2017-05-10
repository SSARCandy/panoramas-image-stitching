# coding: utf-8

import numpy as np

# find best shift using RANSAC
def RANSAC(matched_pairs):
    matched_pairs = np.asarray(matched_pairs)
    
    best_shift = []
    K = 500
    threshold_distance = 20
    
    max_inliner = 0
    for k in range(K):
        
        # Random pick a pair of matched feature
        random_idx = int(np.random.random_sample()*len(matched_pairs))
        sample = matched_pairs[random_idx]
        
        # fit the warp model
        shift = sample[1] - sample[0]
        #print(sample)
        #print(shift)
        
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
def blending(img1, img2, shift):
    blended_img = []
    
    # img2 at lower-left
    if shift[0] > 0 and shift[1] > 0:
        shifted_img = np.lib.pad(img1, [(shift[0], 0), (shift[1], 0), (0, 0)], 'constant', constant_values=0)
        h, w, _ = img2.shape
        _, width, _ = img2.shape

        for y in range(h):
            for x in range(w):
                if list(shifted_img[y][x]) != [0, 0, 0]:
                    color1 = shifted_img[y][x]
                    color2 = img2[y][x]
                    ratio = ((width - x)/shift[1])**3
                    shifted_img[y][x] = (1-ratio)*color1 + ratio*color2
                else:
                    shifted_img[y][x] = img2[y][x]
        blended_img = shifted_img
  
    return blended_img