"""
2018-Q1:    Modified to make it generic & re-wrote some logic to make it function.
@author:    kedar m
"""

import os
import string
import cv2
import xml.etree.ElementTree as ET
import shutil
import numpy as np
import sys
import argparse
import json

# Given a bunch of folders containing images and annotations, copy the image files into
# a single 'Data' folder, and copy the annotation files into a single 'Annotations'
# folder.  Rename the files to insure that filenames can't be duplicated. 
# Create a text file that lists the paths of the merged files.

# Get the Caffe root, COCO Root and Data directories from environmental variables
# If they do not exist, bailout
cocoRoot = os.environ.get("COCO_ROOT")
if (cocoRoot is None) :
    print 'Need to setup enviromental variable COCO_ROOT to proceed'
    raise Exception ('Need to setup enviromental variable COCO_ROOT to proceed')
else :
    print 'COCO_ROOT : {}'.format(cocoRoot)

caffeRoot = os.environ.get("CAFFE_ROOT")
if (caffeRoot is None) :
    print 'Need to setup enviromental variable CAFFE_ROOT to proceed'
    raise Exception ('Need to setup enviromental variable CAFFE_ROOT to proceed')
else :
    print 'CAFFE_ROOT : {}'.format(caffeRoot)

dataRoot = os.environ.get("ML_TEST_DATA_SET_ROOT")
if (dataRoot is None) :
    print 'Need to setup enviromental variable ML_TEST_DATA_SET_ROOT to proceed'
    raise Exception ('Need to setup enviromental variable ML_TEST_DATA_SET_ROOT to proceed')
else :
    print 'ML_TEST_DATA_SET_ROOT : {}'.format(dataRoot)

modelsRoot = os.environ.get("FLIR_MODELS_ROOT")
if (modelsRoot is None) :
    print 'Need to setup enviromental variable FLIR_MODELS_ROOT to proceed'
    raise Exception ('Need to setup enviromental variable FLIR_MODELS_ROOT to proceed')
else :
    print 'FLIR_MODELS_ROOT : {}'.format(modelsRoot)


pythonPath = os.environ.get("PYTHONPATH")
if (pythonPath is None) :
    print 'Need to setup enviromental variable PYTHONPATH to proceed'
    raise Exception ('Need to setup enviromental variable PYTHONPATH to proceed')
else :
    print 'PYTHONPATH : {}'.format(pythonPath)

# read input arguments and setup the required context
parser = argparse.ArgumentParser()
parser.add_argument("dataSetDir", type=str,
                    help="Source Directory of Data Set to be used. Relative Path from ML_TEST_DATA_SET_ROOT")
parser.add_argument("labelMapFile", type=str,
                    help="LabelMapFile to use to convert category names to IDs")

parser.add_argument("valSplit", type=int, default=0, choices=range(5,99),
                    help="Validation Dataset used while training in percentage: 5...99")

args = parser.parse_args()
dataSrcDir = args.dataSetDir
labelMapFile = args.labelMapFile
valSplit = args.valSplit

# setup src dir as absolute path
srcDir = '%s/%s'%(dataRoot, dataSrcDir)
dstDir = srcDir
trainDir = '%s/train'%(dstDir)
valDir = '%s/val'%(dstDir)

print("SrcDir is : " + srcDir)
print("Val Split % is : " + str(valSplit))
print("Train Dir is : " + trainDir)
print("Val Dir is : " + valDir)

# delete all existing (old) directories including sub-dirs first
if os.path.exists(trainDir):
    shutil.rmtree(trainDir)
    print('removed ' + trainDir)

if os.path.exists(valDir):
    shutil.rmtree(valDir)
    print('removed ' + valDir)

# setup passed labelmap file and load label map json data
lmfname = '%s'%(labelMapFile)
print("LabelMapFile to use is : " + lmfname)

with open(lmfname) as lmfile:
    lmdata = json.load(lmfile)

# (km) getCocoID should return a string to serialize the write into XML file
def getCocoID(pascalLabel):

    index = 0
    #debug_count = 0

    for item in lmdata["values"] :
        
        # XXX: Enable this if you are using dataset that has drone tagged as 'cow:20'
        if (pascalLabel == 'cow'):
            return '79';


        if (item == pascalLabel) :
            #print 'pascal label: ' + pascalLabel + 'item: ' + item + 'index: ' + str(index)
            return str(index)

        index += 1

    print pascalLabel + ' not found'
    return 'not found'


# split training and val indices by creating validation dataset index 
# Need to be deterministic i.e., Split should result in the same set of images
#       every time the script is called with other varibles being constant.
def create_test_idx(num_total_samples, percentage):
    num_test_samples = int(num_total_samples*percentage*0.01)
    if (num_test_samples < 1):
        num_test_samples = 1
    np.random.seed(0)
    indices = np.random.permutation(num_total_samples)
    vtest_idx = indices[:num_test_samples]
    #print '{}'.format(vtest_idx)
    return vtest_idx


# Convert floating point bounding box values to int,
# clip the bounding boxes to fit within the image bounds
# and make the category name lower case.
def repairXml(xml_dst, xml_src):
    tree = ET.parse(xml_src)
    root = tree.getroot()

    size = root.find('size')
    width = int(float(size.find('width').text))
    height = int(float(size.find('height').text))

    count = 0

    # convert floating point values in bndbox to int
    for bndbox in root.iter('bndbox'):

        count += 1

        xmin = bndbox.find('xmin')
        val = int(float(xmin.text))
        if (val < 0):
            val = 0
        xmin.text = str(val)

        xmax = bndbox.find('xmax')
        val = int(float(xmax.text))
        if (val < 0):
            val = 0
        if (val > width-1):
            val = width-1
        xmax.text = str(val)

        ymin = bndbox.find('ymin')
        val = int(float(ymin.text))
        if (val < 0):
            val = 0
        ymin.text = str(val)

        ymax = bndbox.find('ymax')
        val = int(float(ymax.text))
        if (val < 0):
            val = 0
        if (val > height - 1):
            val = height -1
        ymax.text = str(val)


    for name in root.iter('name'):
        name.text = name.text.lower()
        # get cocoID for class labels i.e., pascal string to coco integer value
        name.text = getCocoID(name.text)
    
    # put names before bndbox in xml, to workaround bug in caffe
    for object in root.iter('object'):
        name = object.find("name")
        object.remove(name)
        object.insert(0, name)


    # enable this for debug of which files
    #if (count == 0):
    #    print(xml_src + ' : no annotations found')

    if (count > 0):
        tree.write(xml_dst)

    return count


def list_files(path):
    # type: (object) -> object
    # returns a list of names (with extension, and full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(path + name)
    return files

anno = []

# Create a list of folders, each containing xml annotations.  Each annotation
# should have a corresponding image file.  By convention, image and annotation
# files have the same name, only the extension is different.  ie image0.jpg and image0.xml.

# Normalized to Conservator directory structure i.e., Annotations & Data directories
# searches for a directory named 'Annotations' - only two levels deep for now
for name in os.listdir(srcDir):
    srcDir2 = srcDir + '/' + name + '/'
    #if (os.path.isdir(srcDir2)) & (name == 'Annotations'):
    if name == 'Annotations':
        anno.append(srcDir2)
    elif os.path.isdir(srcDir2):
        for name2 in os.listdir(srcDir2):
            srcDir3 = srcDir2 + '/' + name2 + '/'
            if name2 == 'Annotations':
                anno.append(srcDir3)
            else:
                #print('ignoring: ' + srcDir3)
                pass
    else:
        pass

# delete all existing (old) directories including sub-dirs first
if os.path.exists(trainDir):
    shutil.rmtree(trainDir)
    print('removed %s' + trainDir)

if os.path.exists(valDir):
    shutil.rmtree(valDir)
    print('removed %s' + valDir)

# setup train dataset
if not os.path.exists(trainDir):
    os.makedirs(trainDir)
if not os.path.exists(trainDir + '/Annotations/'):
    os.makedirs(trainDir + '/Annotations/')
if not os.path.exists(trainDir + '/Data/'):
    os.makedirs(trainDir + '/Data/')

# setup validation dataset
if not os.path.exists(valDir):
    os.makedirs(valDir)
if not os.path.exists(valDir + '/Annotations/'):
    os.makedirs(valDir + '/Annotations/')
if not os.path.exists(valDir + '/Data/'):
    os.makedirs(valDir + '/Data/')

# We are creating...output files 
#       trainList.txt file: 
#       valList.txt file: 
#               : This has filename.{jpeg,xml} pairs for the given dataset
#               : This file is used by create_data.sh to create the lmdb data files from dataset
#
#       valImageSet.txt file: 
#               : This has a list of all the jpeg files without the extension
#               : This file is used by getImageSize.sh (get_image_size.py), to create the 
#               :       name_size_file text file. Changed get_image_size.py to actually
#               :       read valImageList.txt and get the required info. So, the creation of
#               :       this txt file can be removed.
#
#       valImageList.txt file : This has a list of all the jpeg files
#               : This has a list of all the jpeg files with the extension

# Opening the destination val & train list.txt files..."
dstFile_trainList = open('%s/trainList.txt'%(trainDir), 'w')
dstFile_trainImageList = open('%s/trainImageList.txt'%(trainDir), 'w')
dstFile_trainImageSet = open('%s/trainImageSet.txt'%(trainDir), 'w')

dstFile_valList = open('%s/valList.txt'%(valDir), 'w')
dstFile_valImageList = open('%s/valImageList.txt'%(valDir), 'w')
dstFile_valImageSet = open('%s/valImageSet.txt'%(valDir), 'w')

total_num = 0
for i in range(0, len(anno)):
    list = list_files(anno[i])
    total_num = total_num + len(list)

print('Total Number of Files in Annotations Directories:',total_num)

# valSplit is an input argument
vtest_idx = create_test_idx(total_num, valSplit)
print('Number of Images in the val data set is: ',len(vtest_idx))

# Names are based on consecutive numbers.
newname = 1
numOfFilesWithNoAnnotations = 0
numOfAnnotatedFilesWithNoImages = 0
numOfUsefulImageFiles = 0

for i in range(0,len(anno)):
    list = list_files(anno[i])

    for annoName in list:

        #print('annoName in List: ' + annoName)

        annoPath, annoFilename = os.path.split(annoName)
        oldname, extension = os.path.splitext(annoFilename)
        #print('path.split: ' + annoPath + ' ' + annoFilename)
        #print('file split name & extension: ' + oldname + ' ' + extension)

        # replace the rightmost occurrence of 'Annotations' with 'Data'
        srcImg = 'Data/'.join(annoPath.rsplit('Annotations', 1))
        # print('srcImg path is :' + srcImg)

        # check if src image exists
        #srcPng = srcImg + oldname + '.png'
        #srcJpg = srcImg + oldname + '.jpg'
        #srcJpeg = srcImg + oldname + '.jpeg'
        # read a file from the data directory and get the extension
        # Note: assumption is that all files in that directory are the same type
        #listSrc = list_files(srcImg)
        listSrc = os.listdir(srcImg)
        sfname, sfext = os.path.splitext(listSrc[0])
        srcImgFile = srcImg + oldname + sfext

        # srcImgFile = srcImg + oldname + extension
        #print('Src Image file full path is: ' + srcImgFile)

        # check based on list of generated test indices, if this needs to be part of test or train dataset
        check = np.any(vtest_idx == newname)
        if (check == True):
            merge = valDir
            dstFile = dstFile_valList
            dstFileImageList = dstFile_valImageList
            dstFileImageSet = dstFile_valImageSet
        else:
            merge = trainDir
            dstFile = dstFile_trainList
            dstFileImageList = dstFile_trainImageList
            dstFileImageSet = dstFile_trainImageSet

        # keep the original name as is, and append the index(newname) to make it unique
        # also: don't need to know the extensions. replace this code
        dstImg = merge + '/Data/'
        # dstPng = dstImg + oldname + '-' + str(newname) + '.png'
        # dstJpg = dstImg + oldname + '-' + str(newname) + '.jpg'
        # dstJpeg = dstImg + oldname + '-' + str(newname) + '.jpeg'
        dstImgFile = dstImg + oldname + '-' + str(newname) + sfext
        dstAnno = merge + '/Annotations/' + oldname + '-' + str(newname) + '.xml'

        #print('Dst Image File Full path is: ' + dstImgFile)
        #print(' DstAnno is: ' + dstAnno)
        #print(' annoName is: ' + annoName)

        newname += 1

        annoCount = repairXml(dstAnno, annoName)
        if (annoCount > 0):

            # do it once : NOTE - saving full path name of the file
            #dstAnno = dstAnno.replace(merge, '', 1)

            # do with only one src/dst file
            if os.path.isfile(srcImgFile):
                img = cv2.imread(srcImgFile)
                cv2.imwrite(dstImgFile, img)

                # setup the file name accordingly
                # dstImgFile = dstImgFile.replace(merge, '', 1)
                dstFile.write(dstImgFile)
                dstFile.write(" ")
                dstFile.write(dstAnno)
                dstFile.write("\n")

                # NOTE: (km) Creating imageSet.txt file also here
                imgfname, imgfext = os.path.splitext(dstImgFile)
                dstFileImageSet.write(imgfname)
                dstFileImageSet.write("\n")

                # NOTE: (km) Creating imageList.txt file also here
                dstFileImageList.write(dstImgFile)
                dstFileImageList.write("\n")

                numOfUsefulImageFiles += 1

            #elif os.path.isfile(srcJpg):
            #    print('src file is jpg: ' + srcJpg)
            #    img = cv2.imread(srcJpg)
            #    cv2.imwrite(dstJpg, img)
            #    dstJpg = dstJpg.replace(merge, '', 1)
            #    dstAnno = dstAnno.replace(merge, '', 1)

            #    dstFile.write(dstJpg)
            #    dstFile.write(" ")
            #    dstFile.write(dstAnno)
            #    dstFile.write("\n")
            #
            #elif os.path.isfile(srcJpeg):
            #    print('src file is jpeg : ' + srcJpeg)
            #    img = cv2.imread(srcJpeg)
            #    cv2.imwrite(dstJpeg, img)
            #    dstJpeg = dstJpeg.replace(merge, '', 1)
            #    dstAnno = dstAnno.replace(merge, '', 1)
            #    dstFile.write(dstJpeg)
            #    dstFile.write(" ")
            #    dstFile.write(dstAnno)
            #    dstFile.write("\n")

            else:
                numOfAnnotatedFilesWithNoImages += 1

        else:
            numOfFilesWithNoAnnotations += 1

dstFile.close()
dstFile_trainList.close()
dstFile_trainImageSet.close()
dstFile_trainImageList.close()
dstFile_valList.close()
dstFile_valImageSet.close()
dstFile_valImageList.close()

print(' Number of Files with No Annotations are: ' + str(numOfFilesWithNoAnnotations) + '\n')
print(' Number of Annotated Files with no Images are: ' + str(numOfAnnotatedFilesWithNoImages) + '\n')
print(' Number of Useful Files are: ' + str(numOfUsefulImageFiles))
