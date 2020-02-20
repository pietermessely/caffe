import glob
import cv2
import numpy as np
import random
import shutil
import os

'''

    Q2'2018: (km)
        - Added input arguments & cosmetic changes for generalization

'''

width = 1024
height = 768
channels = 3
numOfImages = 5
dstDir = '/tmp/bgSet1'
debugView = False
np.random.seed(20)

## ================================================================
#
#       Add input arguments
#
## =================================================================

def getInputArgs(argv) :
    
    inOptions = {}
    while argv: 
        if argv[0][0] == '-':
            inOptions[argv[0]] = argv[1]
        argv = argv[1:]

    return inOptions

    
# add grayscale and color noise to the image.  Apply a bit of gaussian blur.
def addNoise(img, amt):

    # random amount of noise between 0 and amt
    variance = random.randint(0, amt)

    img = img.astype(float)

    # zero mean,  same noise added to three image channels
    b, g, r = cv2.split(img)

    noise = np.zeros(shape=b.shape)
    cv2.randn(noise, 0, variance)

    # add greyscale noise + color offset
    b = b + noise + np.random.normal(0, .5)
    g = g + noise + np.random.normal(0, .5)
    r = r + noise + np.random.normal(0, .5 )

    img = cv2.merge((b, g, r))

    sigma = random.random() * 2  # sigma randomly distributed between 0 and 2

    # applying gaussian blur to the input image
    img = cv2.GaussianBlur(img, (0, 0), sigma)

    return img

if __name__ == '__main__' :
    
    from sys import argv
    inArgs = getInputArgs(argv)

    if '-w' in inArgs: 
        width = int(inArgs['-w'])

    if '-t' in inArgs: 
        height = int(inArgs['-t'])

    if '-n' in inArgs: 
        numOfImages = int(inArgs['-n'])

    if '-d' in inArgs: 
        dstDir = inArgs['-d']

    if '-v' in inArgs: 
        debugView = inArgs['-v']
        if not debugView is True : 
            debugView = False

    if '-h' in inArgs: 
        print ('Usage: -w width_of_out_image[e.g., 1024]')
        print ('       -t height_of_out_image[e.g., 768]') 
        print ('       -n number_of_images_to_generate [e.g., 10]') 
        print ('       -d absolute path of destiation directory [e.g., /c/bgSet1 ')
        print ('       -v view images [e.g., True or False]')
        print ('       -h help [e.g., 0..999]')


    # remove destination dir if it exists & create clean one
    if os.path.isdir(dstDir) : 
        errorStr = 'Cleaning up destination_dir '+ dstDir 
        print errorStr
        shutil.rmtree(dstDir)

    os.mkdir(dstDir)

    print ('Image Height is set to {}'.format(height))
    print ('Image Width is set to {}'.format(width))
    print ('Number of Images is set to {}'.format(numOfImages))
    print ('Destination Dir is set to {}'.format(dstDir))
    print ('Debug View is set to {}'.format(debugView))

    for i in range(0, numOfImages):

        #w = int(width/8 + 1)
        #h = int(height/8 + 1)
        w = int(width/8)
        h = int(height/8)

        amount = random.randint(0, 128 )

        img = np.zeros(shape=(w, h, 3))

        img = addNoise(img, amount)
        img = cv2.resize(img, (w*2, h*2))

        img = addNoise(img, amount)
        img = cv2.resize(img, (w*4, h*4))

        img = addNoise(img, amount)
        img = cv2.resize(img, (w*8, h*8))

        img[img > 255] = 255
        img = np.array(img, dtype=np.uint8)

        if debugView is True : 
            cv2.imshow('noise', img)
            cv2.waitKey(0)
        else :
            findx = i + 1
            fileName = dstDir + '/noiseImage_' + str(findx) + '.jpeg'
            cv2.imwrite(fileName, img)

    print('DONE! Generated %d Images'%(numOfImages))


