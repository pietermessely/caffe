# Image augmentation using a gaussian blur kernel.

import glob
import cv2
import numpy as np
import random

show = True
write = False

img_lst = glob.glob('Data/' + '*/*.tiff')
#img_lst = glob.glob('/home/jimk/Downloads/2_phantom_drones_no_propellers_1024x768_180310/Data/' + '*/*.tiff')


np.random.seed(0)
iperm = random.sample(range(len(img_lst)), int(len(img_lst) / 2))
 
count = 0
 
for i in iperm:
    img = cv2.imread(img_lst[i])

    sigma = random.random() * 2   # sigma randomly distributed between 0 and 2

    #print(sigma)
	
    # applying gaussian blur to the input image
    output = cv2.GaussianBlur(img, (0, 0), sigma)

    if show:
        cv2.imshow('noise', img)
        cv2.waitKey(0);

        cv2.imshow('noise', output)
        cv2.waitKey(0);

    if write:
        cv2.imwrite(img_lst[i], output)

    count += 1
    print("Processed:", count)
