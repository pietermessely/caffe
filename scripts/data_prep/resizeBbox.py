"""
Q1'2018:   Created
@author:   kedar m

"""

'''
    This will resize bboxes in input json files relative to input height, width params
        and stash them. The input height & width params are the new image size params.

    The algorithm for resizing the bboxes is as follows:
        for each bbox in input file {
            calculate the relative bbox x/y params based on new/old ratios
            save them
        }

    TODO: 
        Using the resized bboxes, calculate the mAP & show it as mAP' (or mAP-Prime)
        Do this only for confidence threshold 0.4 by default
    DONE: Instead of the above, after resizing the bboxes in both the GT & DT, we call
        the existing evaluation routines to do this
'''
import os
import json
import shutil
import string
import sys
import argparse
import cv2 

## gather environmental variables : Assume these exist as it will be called from another script
modelsRoot = os.environ.get("FLIR_MODELS_ROOT")
dataRoot = os.environ.get("ML_TEST_DATA_SET_ROOT")
caffeRoot = os.environ.get("CAFFE_ROOT")

# input arguments 
parser = argparse.ArgumentParser()

parser.add_argument("gtFile", type=str,
                    help="Ground Truth JSON file. Full Path should be provided")
parser.add_argument("ogtFile", type=str,
                    help="Output Ground Truth JSON file. Full Path should be provided")

parser.add_argument("detResultsFile", type=str,
                    help="Detection Results file. Full Path should be provided")
parser.add_argument("odetResultsFile", type=str,
                    help="Out Detection Results file. Full Path should be provided")

parser.add_argument("newHeight", type=float,
                    help="new height of the image - for use in resizing the bboxes")
parser.add_argument("newWidth", type=float,
                    help="new width of the image - for use in resizing the bboxes")

args = parser.parse_args()
annJson = str(args.gtFile)
ogtFile = str(args.ogtFile)
detResultsFile = str(args.detResultsFile)
odetResultsFile = str(args.odetResultsFile)
new_h = args.newHeight
new_w = args.newWidth

print ('IN : gtFile %s \ndetectionResultsFile %s '%(annJson, detResultsFile))
print ('OUT: gtFile %s \ndetectionResultsFile %s '%(ogtFile, odetResultsFile))

## ===================================================================
#
#       Load the relevant GT and DT info, resize the bboxes and area
#               and save them separately
#
## ===================================================================
ann = open(annJson).read()

# keep a dup of the file contents to modify and write into a separate out file
oGt = iGt = json.loads(ann)
inGtAnno = iGt['annotations']
#oGt = json.loads(ann)
outGtAnno = oGt['annotations']

res = open(detResultsFile).read()
Dt = json.loads(res)
outDt = json.loads(res)

# TODO: Need to do this for each file to accommodate different image sizes
#       move/copy this to the detections and gt routine
# get the original image size & setup the aspect ratio of height and width
old_h = float(iGt['images'][0]['height'])
old_w = float(iGt['images'][0]['width'])

aspect_w = float(new_w / old_w)
aspect_h = float(new_h / old_h)

print('New      Height and Width are - %0.2f : %0.2f \n'%(new_h, new_w))
print('Original Height and Width are - %0.2f : %0.2f \n'%(old_h, old_w))
print('aspect ratio of new/old w:h   - %0.2f : %0.2f \n'%(aspect_w, aspect_h))

## =============================================================================
#       routine to resize inbbox,area into outbbox,area based on aspect ratio
## =============================================================================

def recalc_bbox_area(ibbox, obbox, iaspect_w, iaspect_h) :

    xmin = ibbox[0]
    ymin = ibbox[1]
    xmax = ibbox[2]
    ymax = ibbox[3]
    #print ("orig bbox (x1,y1,x2,y2): %.2f %.2f, %.2f %.2f "%(xmin, ymin, xmax, ymax))

    # resize the bbox
    xmin = max(0.0, xmin * iaspect_w)
    xmax = min(new_w, xmax * iaspect_w)    
    ymin = max(0.0, ymin * iaspect_h)
    ymax = min(new_h, ymax * iaspect_h)    
    #print ("resized bbox (x1,y1,x2,y2):   %.2f %.2f, %.2f %.2f "%(xmin, ymin, xmax, ymax))

    obbox[0] = int(xmin) + 1
    obbox[2] = int(xmax) + 1
    obbox[1] = int(ymin) + 1
    obbox[3] = int(ymax) + 1

## ===================================================================
#
#       for each detection, rewrite the bbox and area
#       then, write into passed out detections file
#
## ===================================================================
i=0

for i in range(len(Dt)) :

    iarea = Dt[i]['area']
    obb = ibb = Dt[i]['bbox']
    recalc_bbox_area(ibb, obb, aspect_w, aspect_h)
    oarea = obb[2] * obb[3]
    outDt[i]['area'] = oarea
    outDt[i]['bbox'] = obb

    #print ("Dt: orig bbox (x1,y1,x2,y2):    %.2f %.2f, %.2f %.2f "%(ibb[0],ibb[1],ibb[2],ibb[3]))
    #print ("Dt: resized bbox (x1,y1,x2,y2): %.2f %.2f, %.2f %.2f "%(obb[0],obb[1],obb[2],obb[3]))
    #print ('Dt: ibb area : obb area are - %d : %d \n'%(iarea, oarea))
    
with open(odetResultsFile, 'w') as ofile:
    json.dump(outDt, ofile, indent=4, sort_keys=True, separators=(',',':'))

## =============================================================
#
# For each gt annotation, rewrite the bbox and area
#       and then write into passed out GT file
#
# check if primed GT annotations file exists. If it does, don't regenerate
#       TODO: To check this or not? Will not for now
#       if not os.path.exists(ogtFile) :
## =============================================================

i = 0
if True :

    #print ("Gt: Number of Annotations are: %d "%(len(inGtAnno)))

    for i in range(len(inGtAnno)) :
        obb = ibb = inGtAnno[i]['bbox']
        iarea = inGtAnno[i]['area']
        recalc_bbox_area(ibb, obb, aspect_w, aspect_h)
        oarea = obb[2] * obb[3]
        outGtAnno[i]['area'] = oarea
        outGtAnno[i]['bbox'] = obb

        #print ("Gt: orig bbox (x1,y1,x2,y2):    %.2f %.2f, %.2f %.2f "%(ibb[0],ibb[1],ibb[2],ibb[3]))
        #print ("Gt: resized bbox (x1,y1,x2,y2): %.2f %.2f, %.2f %.2f "%(obb[0],obb[1],obb[2],obb[3]))
        #print ('GT: ibb area : obb area are - %d : %d \n'%(iarea, oarea))

    oGt['annotations'] = outGtAnno

    with open(ogtFile, 'w') as ofile:
        json.dump(oGt, ofile, indent=4, sort_keys=True, separators=(',',':'))

else:
    print ('Ouput GT file %s exists! Not regenerating\n'%(ogtFile))

