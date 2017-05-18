# coding: utf-8

import cv2
import numpy as np
import constant as const

def compute_r(xx_row, yy_row, xy_row, k):
    row_response = np.zeros(shape=xx_row.shape, dtype=np.float32)
    for x in range(len(xx_row)):
        det_M = xx_row[x]*yy_row[x] - xy_row[x]**2
        trace_M = xx_row[x] + yy_row[x]
        R = det_M - k*trace_M**2
        row_response[x] = R

    return row_response

"""
Harris corner detector

Args:
    img: input image
    pool: for multiprocessing
    k: harris corner constant value
    block_size: harris corner windows size

Returns:
    A corner response matrix. width, height same as input image
"""
def harris_corner(img, pool, k=0.04, block_size=2):
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

    corner_response = pool.starmap(compute_r, [(cov_xx[y], cov_yy[y], cov_xy[y], k) for y in range(height)])
            
    return np.asarray(corner_response)
    
"""
Extract descritpor from corner response image

Args:
    corner_response: corner response matrix
    threshlod: only corner response > 'max_corner_response*threshold' will be extracted
    kernel: descriptor's window size, the descriptor will be kernel^2 dimension vector 

Returns:
    A pair of (descriptors, positions)
"""
def extract_description(img, corner_response, threshold=0.01, kernel=3):
    height, width = corner_response.shape

    # Reduce corner
    features = np.zeros(shape=(height, width), dtype=np.uint8)
    features[corner_response > threshold*corner_response.max()] = 255

    # Trim feature on image edge
    features[:const.FEATURE_CUT_Y_EDGE, :] = 0  
    features[-const.FEATURE_CUT_Y_EDGE:, :] = 0
    features[:, -const.FEATURE_CUT_X_EDGE:] = 0
    features[:, :const.FEATURE_CUT_X_EDGE] = 0
    
    # Reduce features using local maximum
    window=3
    for y in range(0, height-10, window):
        for x in range(0, width-10, window):
            if features[y:y+window, x:x+window].sum() == 0:
                continue
            block = corner_response[y:y+window, x:x+window]
            max_y, max_x = np.unravel_index(np.argmax(block), (window, window))
            features[y:y+window, x:x+window] = 0
            features[y+max_y][x+max_x] = 255

    feature_positions = []
    feature_descriptions = np.zeros(shape=(1, kernel**2), dtype=np.float32)
    
    half_k = kernel//2
    for y in range(half_k, height-half_k):
        for x in range(half_k, width-half_k):
            if features[y][x] == 255:
                feature_positions += [[y, x]]
                desc = corner_response[y-half_k:y+half_k+1, x-half_k:x+half_k+1]
                feature_descriptions = np.append(feature_descriptions, [desc.flatten()], axis=0)
                
    return feature_descriptions[1:], feature_positions

"""
Matching two groups of descriptors

Args:
    descriptor1:
    descriptor2:
    feature_position1: descriptor1's corrsponsed position
    feature_position2: descriptor2's corrsponsed position
    pool: for mulitiprocessing
    y_range: restrict only to match y2-y_range < y < y2+y_range

Returns:
    matched position pairs, it is a Nx2x2 matrix
"""
def matching(descriptor1, descriptor2, feature_position1, feature_position2, pool, y_range=10):
    TASKS_NUM = 32

    partition_descriptors = np.array_split(descriptor1, TASKS_NUM)
    partition_positions = np.array_split(feature_position1, TASKS_NUM)

    sub_tasks = [(partition_descriptors[i], descriptor2, partition_positions[i], feature_position2, y_range) for i in range(TASKS_NUM)]
    results = pool.starmap(compute_match, sub_tasks)
    
    matched_pairs = []
    for res in results:
        if len(res) > 0:
            matched_pairs += res

    return matched_pairs

def compute_match(descriptor1, descriptor2, feature_position1, feature_position2, y_range=10):
    matched_pairs = []
    matched_pairs_rank = []
    
    for i in range(len(descriptor1)):
        distances = []
        y = feature_position1[i][0]
        for j in range(len(descriptor2)):
            diff = float('Inf')
            
            # only compare features that have similiar y-axis 
            if y-y_range <= feature_position2[j][0] <= y+y_range:
                diff = descriptor1[i] - descriptor2[j]
                diff = (diff**2).sum()
            distances += [diff]

        sorted_index = np.argpartition(distances, 1)
        local_optimal = distances[sorted_index[0]]
        local_optimal2 = distances[sorted_index[1]]
        if local_optimal > local_optimal2:
            local_optimal, local_optimal2 = local_optimal2, local_optimal
        
        if local_optimal/local_optimal2 <= 0.5:
            paired_index = np.where(distances==local_optimal)[0][0]
            pair = [feature_position1[i], feature_position2[paired_index]]
            matched_pairs += [pair]
            matched_pairs_rank += [local_optimal]

    # Refine pairs
    sorted_rank_idx = np.argsort(matched_pairs_rank)
    sorted_match_pairs = np.asarray(matched_pairs)
    sorted_match_pairs = sorted_match_pairs[sorted_rank_idx]

    refined_matched_pairs = []
    for item in sorted_match_pairs:
        duplicated = False
        for refined_item in refined_matched_pairs:
            if refined_item[1] == list(item[1]):
                duplicated = True
                break
        if not duplicated:
            refined_matched_pairs += [item.tolist()]
            
    return refined_matched_pairs
