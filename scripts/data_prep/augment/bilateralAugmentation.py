# Image augmentation using a gaussian blur kernel.

import glob
import cv2
import numpy as np
import random

show = True 
write = False 

img_lst = glob.glob('Data/*.jpeg')

np.random.seed(0)
 
for i in img_lst:
    img = cv2.imread(i)

    smoothing = 200 * random.random()   # sigma randomly distributed between 0 annd 200 

    print(smoothing)
	
    # applying gaussian blur to the input image
    output = cv2.bilateralFilter(img, 9, smoothing, smoothing)
    if show:
        cv2.imshow('noise', img)
        cv2.waitKey(0);

        cv2.imshow('noise', output)
        cv2.waitKey(0);

    if write:
        cv2.imwrite(i, output)

