
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

    # random amount of noise between 0 and 25
    variance = random.randint(0, 25)

    # zero mean,  same variance in three image channels
    noise = img.copy()
    cv2.randn(noise, 0, (variance, variance, variance))

    # add noise and clamp to 255
    img = img.astype(int)
    imgPlus = img + noise

    imgPlus[imgPlus > 255] = 255

    img = np.array(img, dtype=np.uint8)
    output = np.array(imgPlus, dtype=np.uint8)

    if show:
        cv2.imshow('noise', img)
        cv2.waitKey(0);

        cv2.imshow('noise', output)
        cv2.waitKey(0);

    if write:
        cv2.imwrite(img_lst[i], output)

    count += 1
    print("Processed:", count, i)
	
