# from __future__ import print_function

"""
Q2'2018:   Created
@author:   kedar m

"""

'''
    This script will determine the visibility of the object in each bounding box.
        It will remove bounding boxes that have low visibility and ouput a new json file

    Mandatory* and Optional# Arguments & Outputs:
    -------------------------------
        a*. Absolute path of source image  (e.g., /mnt/data/dataset/rgb/003000_rgb.jpg)
        b*. Absolute path of 'segmentation" image  (e.g., /mnt/data/dataset/segment/003000_segment.png)
        c*. Absolute path of the source json file (e.g., /mnt/data/dataset/rgb/003000_rgb.json)
        d*. Absolute path of the destination json file (e.g., /mnt/data/datasetFiltered/rgb/003000_rgb.json)
        e#. colorBW : color for foreground images is black/white(1) or multi-color (0)
        f#. view : view final images and bounding boxes (1) or save them in the dst dir (0)
 
'''
import os
import json
import shutil
import string
import sys
import argparse
import random
import glob
import numpy as np
import cv2
import json
from pprint import pprint
import matplotlib.pyplot as plt


## ================================
# Defaults
## =================================

random.seed(100)

debugView = 0  # default to disable i.e., will write the file

histVals = []   # a place to accumulate drone area values


## ================================================================
#       Input Arguments
## =================================================================

def getInputArgs(argv):
    inOptions = {}
    while argv:
        if argv[0][0] == '-':
            inOptions[argv[0]] = argv[1]
        argv = argv[1:]

    return inOptions

## ================================================================
#       Compute and plot a histogram of areas
## =================================================================
def area_histogram():

    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(histVals, num_bins, density=1)

    ax.set_xlabel('drone bbox size (area in pixels)')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

## =============================================
#  visibility of pixels under the mask
## =============================================

def compute_visibility(roiImg, roiMask):

    inv_mask = cv2.bitwise_not(roiMask)

    meanTarget = cv2.mean(roiImg, roiMask)[0]
    meanBackground = cv2.mean(roiImg, inv_mask)[0]

    return abs(meanTarget - meanBackground) / 255.0

## =============================================
#       using segmentation masks of fgImage
## =============================================


def similarity_filter(image_path, seg_image_path, json_src, json_dst, debug_view, vis_threshold, area_threshold):
    #   Read the Image and get image sizes. Ensure same size for all.
    global total_count
    global rejection_count

    dst_annotation = {}
    dst_annotation["annotation"] = []

    img = cv2.imread(image_path)
    segImg = cv2.imread(seg_image_path)

    # fh,fw = fimg.shape[:2]
    ih, iw = img.shape[:2]
    sh, sw = segImg.shape[:2]

    if (ih != sh) or (iw != sw):
        print 'image and segment image aspect ratios are not the same. Aborting'
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seg_image_gray = cv2.cvtColor(segImg, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(seg_image_gray, 64, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)

    data = json.load(open(json_src))

    filtered_data = json.load(open(json_src))
    filtered_data["annotation"] = [] # we will replace it with filtered annoations

    for annotation in data["annotation"]:

        xmin = int(annotation["bbox"][0])
        ymin = int(annotation["bbox"][1])
        xmax = int(annotation["bbox"][2]) + xmin
        ymax = int(annotation["bbox"][3]) + ymin

        imgRoi = img_gray[ymin: ymax, xmin: xmax]
        maskRoi = mask[ymin: ymax, xmin: xmax]

        visibility = compute_visibility(imgRoi, maskRoi)

        total_count += 1
        if visibility > vis_threshold and annotation["area"] > area_threshold:
            filtered_data["annotation"].append(annotation)
            #del data["annotation"][count]
            rejection_count += 1
        #else:
            #count += 1

        histVals.append( annotation["area"])

        if debug_view:

            if (visibility < vis_threshold or annotation["area"] < area_threshold):
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))

            # cv2.imshow("img", segImg)
            # cv2.waitKey(0)
            #
            # cv2.imshow("roi", imgRoi)
            # cv2.waitKey(0)
            #
            # cv2.imshow("roi", maskRoi)
            # cv2.waitKey(0)

            cv2.imshow("img", img)

        print('visibility: ' + str(visibility) + ' area: ' +   str(annotation["area"]))

    tmp_path = 'drone_apr20/tmp/' + os.path.basename(image_path)
    tmp_path = tmp_path.replace('rgb', 'tmp')

    #tmp_path = 'tmp/{0:03d}.jpg'.format(count)
    #cv2.imwrite(tmp_path, img)
    cv2.waitKey(1)


    if (not debug_view):
        with open(json_dst, 'w') as outfile:
            json.dump(filtered_data, outfile, indent=4, sort_keys=True)


def main(args):
    from sys import argv

    inArgs = getInputArgs(argv)

    if '-i' in inArgs:
        image_path = inArgs['-i']
    else:
        raise ValueError('No source image provided')

    if '-s' in inArgs:
        seg_image_path = inArgs['-s']
    else:
        raise ValueError('No segmentation image  provided')

    if '-j' in inArgs:
        src_json_path = inArgs['-j']
    else:
        raise ValueError('No json annotation file provided')

    if '-o' in inArgs:
        dst_json_path = inArgs['-o']
    else:
        raise ValueError('No output (filtered annotation file) path provided')

    if '-p' in inArgs:
        overlayFileNamePrefix = inArgs['-p']

    if '-v' in inArgs:
        debug_view = int(inArgs['-v'])
    if not debug_view is 1:
        debug_view = 0

    if '-c' in inArgs:
        useBWcolors = int(inArgs['-c'])

        if not useBWcolors is 1:
            useBWcolors = 0
            print 'Foreground Images will be of various colors'
            print 'numOfPreDefinedColors & Max Colors are: %d, %d ' % (numOfPreDefinedColors, maxColors)
        else:
            print 'Foreground Images will only use black / white colors'
            numOfPreDefinedColors = len(preDefinedColorsBW)
            maxColors = numOfPreDefinedColors
            print 'numOfPreDefinedColors & Max Colors are: %d, %d ' % (numOfPreDefinedColors, maxColors)

    if '-h' in inArgs:
        print ('Usage: -i input Image full path [e.g., /tmp/dataset/rgb/0001.jpg]')
        print ('       -s segmentation Images Dir full path [e.g., /tmp/dataset/segmentation/0001.jpg]')
        print ('       -j json annotation file [e.g., /tmp/dataset/rgb/0001.json]')
        print ('       -o output filtered json annotation file')
        print ('       -h help [e.g., 0..100 - print help')
        print ('       -v view images [e.g., True or False]')

    ## =============================================
    #   Call the similarityFilter routine
    ## =============================================

    #similarity_filter(image_path, seg_image_path, src_json_path, dst_json_path, debug_view)

if __name__ == '__main__':
    main(sys.argv[1:])

####################################################################################
### TODO: How do i move this to a python script that calls similarity_filter()?

total_count = 0
rejection_count = 0

img_list = glob.glob('/mnt/data/droneDatasets/drone_apr20/rgb/*.jpg')

for img_path in img_list:

    seg_img_path = img_path.replace('rgb', 'segmentation')
    src_json_path = img_path.replace('.jpg', '.json')

    dst_json_path = 'drone_apr20/filtered/' + os.path.basename(src_json_path)

    similarity_filter(img_path, seg_img_path, src_json_path, dst_json_path, 0, 0.1, 15*15)

#print ('rejected ' + rejection_count + ' out of ' + total_count + str(rejection_count/float(total_count)) +  ' percent.')
print total_count
print rejection_count

area_histogram()