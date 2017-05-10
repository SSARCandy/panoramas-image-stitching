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
    padding = [
        (shift[0] if shift[0] > 0 else 0, -shift[0] if shift[0] < 0 else 0),
        (shift[1] if shift[1] > 0 else 0, -shift[1] if shift[1] < 0 else 0),
        (0, 0)
    ]
    shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)
    
    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape
    
    inv_shift = [h1-h2, w1-w2]
    inv_padding = [
        (inv_shift[0] if shift[0] < 0 else 0, inv_shift[0] if shift[0] > 0 else 0),
        (inv_shift[1] if shift[1] < 0 else 0, inv_shift[1] if shift[1] > 0 else 0),
        (0, 0)
    ]
    shifted_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)

    for y in range(h1):
        for x in range(w1):
            color1 = shifted_img1[y][x]
            color2 = shifted_img2[y][x]
            
            if list(color1) == [0, 0, 0]:
                shifted_img1[y][x] = color2
            elif list(color2) == [0, 0, 0]:
                shifted_img1[y][x] = shifted_img1[y][x]
            else:
                ratio = x/w1
                if ((color1 - color2)**2).sum() > 10**2:
                    ratio = 1
                shifted_img1[y][x] = (1-ratio)*color1 + ratio*color2
      
    return shifted_img1