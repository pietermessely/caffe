# Image augmentation using a 'motion blur' kernel.  Convolve with one of the following where a == 1.0 / size

#   a 0 0 0 0
#   0 a 0 0 0
#   0 0 a 0 0
#   0 0 0 a 0
#   0 0 0 0 a

#   0 0 0 0 a
#   0 0 0 a 0
#   0 0 a 0 0
#   0 a 0 0 0
#   a 0 0 0 0

#   0 0 0 0 0
#   0 0 0 0 0
#   a a a a a
#   0 0 0 0 0
#   0 0 0 0 0


import glob
import cv2
import numpy as np
import random

show = True
write = False

#img_lst = glob.glob('/home/jimk/Downloads/2_phantom_drones_no_propellers_1024x768_180310/Data/' + '*/*.tiff')
img_lst = glob.glob('Data/' + '*/*.tiff')

np.random.seed(0)

iperm = random.sample(range(len(img_lst)), int(len(img_lst) / 2))

size = 7
radius = int(size/2)

# horizontal motion kernel
motion_horz = np.zeros((size, size))
motion_horz[radius, ...] = 1.0 / size

# diagonal motion kernels
motion_diag = np.zeros((size, size))
np.fill_diagonal(motion_diag, 1.0/size)

motion_diag_flip = np.flip(motion_diag, 1) # left/right flip

count = 0
for i in iperm:

    img = cv2.imread(img_lst[i])

    r = random.randint(0, 3)

    # applying the kernel to the input image
    if r < 2 :    # 50% probability of horizontal blur
        output = cv2.filter2D(img, -1, motion_horz)
    if r == 2:
        output = cv2.filter2D(img, -1, motion_diag)
    if r == 3:
        output = cv2.filter2D(img, -1, motion_diag_flip)

    if (show):
        cv2.imshow('blur', img)
        cv2.waitKey(0);

        cv2.imshow('blur', output)
        cv2.waitKey(0)

    if write:
        cv2.imwrite(img_lst[i], output)

    count += 1
    print("Processed:", count, i)
	
