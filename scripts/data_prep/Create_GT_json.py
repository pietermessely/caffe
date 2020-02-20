"""
Q2'2018:   create additional GT files for small, medium and large objects
@author:   kedar m

Q1'2018:   Augmented to count size of each label/object as s/m/l in the dataset and
            write into a text file under the 'test' directory.
@author:   kedar m

Q4'2017:   Modified parts to be generic and rewrote some logic to make it function
@author:   kedar m
"""

'''
    Creating a GT json file from modified xml files.
    The xml file mods are done using createList.py.

    This routine takes input arguments to get the dataset dir and
    type of dataset usage i.e., for test(1) or train_val(0).

    This routine also reads environment variables to setup some contexts.
    If it doesn't find what it is looking for, it will barf.

'''

import os
import xml.etree.cElementTree as ET
import skimage.io as io
import numpy as np
#import matplotlib.pyplot as plt
import json
from random import shuffle
import shutil
import argparse

# src json annotation to dst json
def json2json(dstFile, srcFile, imgFile, imageId):

    I = io.imread(imgFile)
    imgHeight = I.shape[0]
    imgWidth =  I.shape[1]

    # setup input file and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    image = idata["image"]

    # update image data
    image["file_name"] = imgFile
    image["id"] = imageId
    image["height"] = imgHeight
    image["width"] = imgWidth

    #change annotation data
    anns = idata["annotation"]
    for d1 in range(len(anns)) :
        anns[d1]["image_id"] = imageId

    ofile = open(dstFile, 'w')
    json.dump(idata, ofile, indent=4, sort_keys=True, separators=(',',':'))


# src xml annotation to dst json
def xml2json(dstFile, xmlFile, imgFile, image_id):

    anno_id = 0

    I = io.imread(imgFile)

    imgHeight = I.shape[0]
    imgWidth =  I.shape[1]

    # img = {'file_name': 'img{}.jpg'.format(image_id), 'height': imgHeight, 'id': image_id, 'width': imgWidth}
    img = {'file_name': '{}'.format(imgFile), 'height': imgHeight, 'id': image_id, 'width': imgWidth}
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    for child in root:

        anns = []

    for object in root.findall('object'):

        bndbox = object.find('bndbox')

        # (km) 3/20 - if there are no bounding boxes, continue to accommodate files
        #               with no annotations
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        x = xmin - 1
        y = ymin - 1

        # clamp values to image bounds
        x = max(0, x)
        y = max(0, y)
        xmax = min(imgWidth-1, xmax)
        ymax = min(imgHeight-1, ymax)

        width = (xmax - xmin) + 1
        height = (ymax - ymin) + 1

        area = width * height
        bbox = [ x, y, width, height ]

        # (km)
        # 	don't convert name string to number yet. We will do that later.
        #
        category_str = object.find('name').text
        category_id = int(category_str)
        if (category_id < 0):
            continue

        segmentation=[x, y, x, ymax, xmax, ymax, xmax, y]

    	# anns.append( { 'area': area, 'bbox': bbox, 'category_id': category_id, 'id': anno_id, 'ignore': 0, 'image_id': image_id,  'iscrowd': 0, 'segmentation': [segmentation] })

        anns.append( { 'area': area, 'bbox': bbox, 'category_id': category_id, 'id': anno_id,  'image_id': image_id,  'iscrowd': 0, 'segmentation': [segmentation] })

        anno_id = anno_id + 1

    data = {  'annotation': anns, 'image': img }

    with open(dstFile, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, separators=(',', ':'))

#
# setup contexts from environment variable and input arguments
#
dataRoot = os.environ.get("ML_TEST_DATA_SET_ROOT")
#modelsRoot = os.environ.get("FLIR_MODELS_ROOT")
if (dataRoot is None) :
    print 'Need to setup enviromental variable ML_TEST_DATA_SET_ROOT to proceed'
    raise Exception ('Need to setup enviromental variable ML_TEST_DATA_SET_ROOT proceed')
else :
    print 'ML_TEST_DATA_SET_ROOT : {}'.format(dataRoot)

#if (modelsRoot is None) :
#    print 'Need to setup enviromental variable FLIR_MODELS_ROOT to proceed'
#    raise Exception ('Need to setup enviromental variable FLIR_MODELS_ROOTto proceed')
#else :
#    print 'FLIR_MODELS_ROOT : {}'.format(modelsRoot)

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataSetDir", type=str,
                    help="Source Directory of Data Set to be used. Relative Path from ML_TEST_DATA_SET_ROOT")
parser.add_argument("isDataSetUsedForValidation", type=int, choices=range(0,2),
                    help="Data Set is used for Train+val(0) or for test(1)")
parser.add_argument("labelFile", type=str,
                    help="Name of labelfile to be used : FULL PATH")
parser.add_argument("gtFile", type=str,
                    help="Name of gt file to be used : FULL PATH")

args = parser.parse_args()
dataSrcDir = args.dataSetDir
dataType = int(args.isDataSetUsedForValidation)
labelFile = args.labelFile
gtFile = args.gtFile

valDS = 'val'
testDS = 'test'
trainDS = 'train'

#print ("dataSrcDir: %s, dataType: %s"%(dataSrcDir, dataType))

#baseDir = '/home/flir/warea/ub-setup/caffe/data/aporia-cars-nov9/val'
#baseDir = '/home/flir/warea/ub-setup/caffe/data/dronedata/val'
baseDir = '%s/%s'%(dataRoot,dataSrcDir)
valDir = '%s/%s'%(baseDir,valDS)
testDir = '%s/%s'%(baseDir,testDS)
trainDir = '%s/%s'%(baseDir,trainDS)

#
# setup required directory variables based on input information
# NOTE: These are defaults based on conservator dataset hierarchy and image file extensions
#
#xmlDir = '%s/Annotations/'%(baseDir)
#jsonDir = '%s/json/'%(baseDir)
#dataDir = '%s/Data/'%(baseDir)

if (dataType == 1) :

    # test data set
    xmlDir = '%s/Annotations/'%(testDir)
    jsonDir = '%s/json/'%(testDir)
    dataDir = '%s/Data/'%(testDir)

else:

    # train+val data set
    raise Exception ('Train & Val dataset is NOT supported')

# read a file from the data directory and get the extension
# Note: assumption is that all files in that directory are the same type
# TODO: check & read until a valid Type is found i.e., .jpg or .jpeg or .tiff or .png
listSrc = os.listdir(dataDir)
nullname, imgType = os.path.splitext(listSrc[0])


#print ("valDir: " + valDir)
print ("testDir: " + testDir)
print ("labelFile: " + labelFile)

# empty json folder to fill up with new json files
if os.path.exists(jsonDir):
	shutil.rmtree(jsonDir)
	os.makedirs(jsonDir)
else:
	os.makedirs(jsonDir)

# convert each xml to its corresponding json or if json src, just update and copy
srcIsJson = False
for infile in os.listdir(xmlDir):
	filename, file_extension = os.path.splitext(infile)

        # support .json annotations also
        if file_extension == '.json' :
            jsonFile = infile
            srcIsJson = True
        else :
	    jsonFile = filename + '.json'
            srcIsJson = False

	dataFile = filename + imgType
	srcFile = xmlDir + infile
	imgFile = dataDir + dataFile
	dstFile = jsonDir + jsonFile

	#
	# conservator filename is : video-D6YuLc44qf4k62YWn-frame-000140-sQcRbCiwHfr6joJXY
	# we augment this with a unique number in new-createList.py
        #          video-D6YuLc44qf4k62YWn-frame-000140-sQcRbCiwHfr6joJXY-12345
        # Let us get the imageId without needing to hard code to a specific format
        #image_id = int(filename.split('-')[5]) ## hardcoded to above file format
        image_id = int(filename.split('-')[-1])
	#print(image_id)

        if srcIsJson is False :
	    xml2json(dstFile, srcFile, imgFile, image_id)
        else :
            json2json(dstFile, srcFile, imgFile, image_id)


# read the labelmap.json file and create the category list for GT file
with open(labelFile, 'r') as ifile:
    lmdata = json.load(ifile)

catdata = []
i=0
for cat in lmdata["values"] :
    if i == 0 :
        i = 1
        continue

    catdata.append( { 'id' : i, 'name' : str(cat) } )
    i += 1

print '{}: Number of categories: {}'.format(__file__, i)
#print catdata


#       Merge all json files into one big json which is compatible with the GT input
#       format of pycocoEval.py

keys = ['images', 'annotations', 'categories']
Anno = {key: None for key in keys}
AnnoS = {key: None for key in keys}
AnnoM = {key: None for key in keys}
AnnoL = {key: None for key in keys}
lst0 = []
lst1 = []
lstS = []
lstM = []
lstL = []

sarea = 32*32
marea = 96*96

for file in os.listdir(jsonDir):
	fullname = jsonDir + file
	eachJson = open(fullname).read()
	j_data = json.loads(eachJson)

	val0 = j_data.values()[0]
	if isinstance(val0, list):
	    #print('val0 is a list')
	    for d0 in range(len(val0)):
		lst0.append(val0[d0])
	else:
	    lst0.append(val0)

	val1 = j_data.values()[1]
	if isinstance(val1, list):
	    #print('val1 is a list')
	    for d1 in range(len(val1)):
	        lst1.append(val1[d1])
                area = int(val1[d1]['area'])
                if area < sarea :
                    lstS.append(val1[d1])
                if area < marea :
                    lstM.append(val1[d1])
                else :
                    lstL.append(val1[d1])
	else:
	    lst1.append(val1)
            area = int(val1['area'])
            if area < sarea :
                lstS.append(val1)
            if area < marea :
                lstM.append(val1)
            else :
                lstL.append(val1)


Anno['images'] = AnnoS['images'] = AnnoM['images'] = AnnoL['images'] = lst0
Anno['categories'] = AnnoS['categories'] = AnnoM['categories'] = AnnoL['categories'] = catdata
Anno['annotations'] = lst1
AnnoS['annotations'] = lstS
AnnoM['annotations'] = lstM
AnnoL['annotations'] = lstL

## ==========================================================================
# change anno id's
#
# count label/object sizes of each as s, m, l - i.e., for all classes in GT
## ==========================================================================

#numOfLabels = len(lmdata["values"])            ## HACK ALERT XXX
numOfLabels = len(lmdata["values"]) + 1         ## TODO: COCO dataset has catid 90 assigned
sobj = [0 for i in range(numOfLabels)]
mobj = [0 for i in range(numOfLabels)]
lobj = [0 for i in range(numOfLabels)]

num = 0
for i in range(len(Anno.values()[1])):

    Anno.values()[1][i]['id'] = num

    area = int(Anno.values()[1][i]['area'])
    catid = int(Anno.values()[1][i]['category_id'])

    if area < sarea :
        sobj[catid] += 1
    elif area < marea :
        mobj[catid] += 1
    else :
        lobj[catid] += 1

    num = num + 1

## =================================================================
#       save the count of each object in a file
## =================================================================
dstFileSizesList = open('%s/gt_object_size_list.txt'%(testDir), 'w')
i=0
print "num of labels: " + str(numOfLabels)

for i in range(numOfLabels - 1):
    #print lmdata["values"][i]
    if (sobj[i] > 0 or mobj[i] > 0 or lobj[i] > 0) :
        tstr='%s(%d) %d %d %d'%(lmdata["values"][i], i, sobj[i], mobj[i], lobj[i])
        dstFileSizesList.write(tstr)
        dstFileSizesList.write("\n")
    i += 1

dstFileSizesList.close()


# save final GT json into the file name provided (full path)
with open(gtFile, 'w') as f:
    #json.dump(Anno, f)
    json.dump(Anno, f, indent=4, sort_keys=True, separators=(',', ':'))


## =====================================================================
#       create a GT file for each of small, medium, large object sizes
## =====================================================================

gtPath, gtFname = os.path.split(gtFile)
gtFile_small = gtPath + '/small_' + gtFname
gtFile_medium = gtPath + '/medium_' + gtFname
gtFile_large = gtPath + '/large_' + gtFname

#print gtFile_small
#print gtFile_medium
#print gtFile_large

with open(gtFile_small, 'w') as f:
    json.dump(AnnoS, f, indent=4, sort_keys=True, separators=(',', ':'))

with open(gtFile_medium, 'w') as f:
    json.dump(AnnoM, f, indent=4, sort_keys=True, separators=(',', ':'))

with open(gtFile_large, 'w') as f:
    json.dump(AnnoL, f, indent=4, sort_keys=True, separators=(',', ':'))
