import cv2
import numpy as np
import os
import argparse
import sys
import glob
import time
from datetime import timedelta

if __name__ == '__main__':

    # input arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="directory for input images")
    parser.add_argument("--dst_dir", type=str, help="directory for output images")
    args = parser.parse_args()
    inDir = args.src_dir or 'need-an-input-dir'
    outDir = args.dst_dir or 'need-an-output-dir'

    if not os.path.isdir(inDir) : 
        parser.print_help()
        raise ValueError ('Invalid images directory: ' + inDir)

    if not os.path.isdir(outDir) : 
        os.makedirs(outDir)

    imagesList = glob.glob(inDir + '/*')
    print 'Number of Images to process: {}'.format(len(imagesList))

    i = 0
    start_time = time.time()
    for imageFile in imagesList :

        fpath, fname = os.path.split(imageFile)
        dstFile = outDir + '/' + fname

        img = cv2.imread(imageFile)
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        igimg = cv2.bitwise_not(gimg)
        cv2.imwrite(dstFile, igimg)

        if (i % 100) == 0: 
            print str(i)
        i += 1
        
    print '======================================'
    print 'Time Taken: {}'.format(timedelta(seconds=(time.time()-start_time)))
    print '======================================'

################################################################################

'''
Q4-2018 kedar

script to convert images to grayscale and invert.

Needs src_dir and dst_dir as input arguments. Converts all images in the input dir
and saves them in the dst dir.

'''

