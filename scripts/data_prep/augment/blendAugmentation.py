# Image augmentation using a gaussian blur kernel.

import glob
import cv2
import numpy as np
import random

show = True
write = False

img_lst = glob.glob('Data/' + '*/*.tiff')
#img_lst = glob.glob('/home/jimk/Downloads/2_phantom_drones_no_propellers_1024x768_180310/Data/' + '*/*.tiff')
#img_lst = glob.glob('/mnt/data/local/2_Phantom_Drones_Blurred_Propellers_180413_NewMotionBlur_Blend/Data_test/' + '*.jpg')

np.random.seed(0)
count = 0
 
for i in range(len(img_lst)):
    img = cv2.imread(img_lst[i])

    #grascale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
    #create 3-channel grayscale image
    img_gray3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

	#apply with 0-100% blending
    alpha = random.uniform(0.0, 1.0)
    beta = 1.0 - alpha
    output = cv2.addWeighted(img, alpha, img_gray3, beta, 0.0)

    if show:
        cv2.imshow('original', img)
        cv2.waitKey(0);

        cv2.imshow('blended', output)
        cv2.waitKey(0);

    if write:
        cv2.imwrite(img_lst[i], output)

    count += 1
    print("Processed:", count)
