# coding: utf-8

import os
import sys
import cv2


def load_images(source_dir):
    imgs = []
    
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        imgs.extend([filename for filename in filenames if filename.endswith('.jpg') or filename.endswith('.png')])
        break
    
    image_list = [cv2.imread(os.path.join(source_dir, img), 1) for img in imgs]

    return image_list
