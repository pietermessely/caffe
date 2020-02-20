"""
2018-Q2:    Modified to support separate evaluationScript needs
@author:    kedar m
2017-Q4:    Modified to make it generic & re-wrote some logic to make it function.
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
import labelmap_utils as lmutil

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
                    help="LabelMapFile to use to convert category names to IDs. Relative Path from FLIR_MODELS_ROOT")
parser.add_argument("labelVectorFile", type=str,
                    help="labelVectorFile to map ids to.")
#parser.add_argument("imageJsonFileName", type=str,
#                   help="image.json file to use for setting up the data. If NULL, sets up based on dirs")
parser.add_argument("annoJsonFileName", type=str,
                   help="annotation.json file to use for setting up the data. If NULL, should barf!! ")
parser.add_argument("--imageDirPrefix", type=str,
                   help="prefix to put on image filenames. default is 'Data'")
parser.add_argument("--includeAllGt", type=str,
                   help="Include All GT (yes) or only GT with Annotations(no). Default is 'yes'")
parser.add_argument("--remap_labels", type=str, help="JSON file specifying label renaming of annotations to catids as { src_label: dst_label }")


args = parser.parse_args()
dataSrcDir = args.dataSetDir
labelMapFile = args.labelMapFile
labelVectorFile = args.labelVectorFile
#imageJsonFileName = args.imageJsonFileName
annoJsonFileName = args.annoJsonFileName
imageDirPrefix = args.imageDirPrefix or "Data"
includeAllGt = args.includeAllGt or "yes"
includeAllGt = includeAllGt.lower()

# Only supports test dataset usecase.
srcDir = '%s/%s'%(dataRoot, dataSrcDir)
dstDir = srcDir
testDir = '%s/test'%(dstDir)

print("SrcDir is : " + srcDir)
print("testDir is : " + testDir)

# Load label remap
remap_label_to_catid_name = {}
if args.remap_labels:
    print "Loading label remap file: %s" % args.remap_labels
    f = open(args.remap_labels, 'r')
    remap_label_to_catid_name = json.load(f)

## use labelmap file
lmfname = '%s'%(labelMapFile)
lvfname = '%s'%(labelVectorFile)
# Setup dicts to easily go between cocoid, name, and vectorid.
name_to_coco_id = lmutil.get_name_to_coco_catid(lmfname)
name_to_vector_id = lmutil.get_name_to_labelvector_id(lvfname)
coco_id_to_vector_id = {}
for name in name_to_coco_id:
    coco_id_to_vector_id[name_to_coco_id[name]] = name_to_vector_id.get(name, 0)

print("LabelMapFile to use is : " + lmfname)
#print("imageJsonFile to use is : " + imageJsonFileName)
print("annoJsonFile to use is : " + annoJsonFileName)
print("includeAllGt is : " + includeAllGt)
print("imageDirPrefix is : " + imageDirPrefix)


# Convert floating point bounding box values to int,
# clip the bounding boxes to fit within the image bounds
# and make the category name lower case.
def repairXml(xml_dst, xml_src):
    classes_without_coco_labels = []

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
        # Run label through remapping
        name.text = remap_label_to_catid_name.get(name.text, name.text)
        # get labelvector id for class labels i.e., pascal string to vector integer value
        coco_id = name_to_vector_id.get(name.text, -1)
        if (coco_id == -1):
            classes_without_coco_labels.append(name.text)
        name.text = str(coco_id)

    # put names before bndbox in xml, to workaround bug in caffe
    for object in root.iter('object'):
        name = object.find("name")
        if not name is None :
            object.remove(name)
            object.insert(0, name)

    # enable this for debug of which files
    #if (count == 0):
    #    print(xml_src + ' : no annotations found')

    #if includeAllGt == 'yes':
    #    count = 1

    if (count > 0) or (includeAllGt == 'yes') :
        tree.write(xml_dst)

    # useful for debug
    if len(classes_without_coco_labels) > 100:
        print "NOTE: Labeled objects that are not in your label file found in '%s'" % xml_src
        print set(classes_without_coco_labels)

    return count


# ====================================================
# create the new json file from old one and update 'file_name'
#       field of the 'image' node in dst file to passed fileName and
#       update the 'image_id' field in each annotation
# ====================================================
def json2json(srcFile, dstFile, fileName, imageId) :

    #print ('json2json srcFile: '+ srcFile + 'dstFile: ' + dstFile + 'fileName: ' + fileName)
    #print ('imageId: %s'%(imageId))

    # setup input file and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    image = idata["image"]

    # update image data
    image["file_name"] = fileName
    image["id"] = imageId

    anns = idata["annotation"]
    #change annotation data
    for anno in anns :
        anno["image_id"] = imageId
        #print str(anno["category_id"]) + " " + str(coco_id_to_vector_id[int(anno["category_id"])])
        anno["category_id"] = coco_id_to_vector_id[int(anno["category_id"])]

    ofile = open(dstFile, 'w')
    json.dump(idata, ofile, indent=4, sort_keys=True, separators=(',',':'))

    return len(anns)


def list_files(path):
    # returns a list of file names (with extension, and full path)
    # in folder path, and any from sub-directories at one level deep
    files = []
    for name in os.listdir(path):
        fullPathName = os.path.join(path,name)
        if os.path.isdir(fullPathName):
            for fname in os.listdir(fullPathName) :
                fullFname = os.path.join(fullPathName,fname)
                if os.path.isfile(fullFname):
                    files.append(fullFname)
        elif os.path.isfile(fullPathName):
            files.append(fullPathName)
        else:
            pass

    return files


# Create a list of folders, each containing xml annotations.  Each annotation
# should have a corresponding image file.  By convention, image and annotation
# files have the same name, only the extension is different.  ie image0.jpg and image0.xml.

#
# Normalized to Conservator directory naming structure i.e., "Annotations" & "Data" dirs
# searching for a directory named 'Annotations' - only two levels deep for now
#

anno = []

def gatherAnnotationDirs (srcDir) :
    total_num = 0
    for name in os.listdir(srcDir):
        srcDir2 = srcDir + '/' + name + '/'
        #if (os.path.isdir(srcDir2)) & (name == 'Annotations'):
        if name == 'Annotations':
            anno.append(srcDir2)
        elif name == 'test':
            pass
        elif os.path.isdir(srcDir2):
            for name2 in os.listdir(srcDir2):
                srcDir3 = srcDir2 + '/' + name2 + '/'
                if name2 == 'Annotations':
                    anno.append(srcDir3)
                else:
                    pass

    print('Total Number of Annotation Directories: '+ str(len(anno)))
    for i in range(0, len(anno)):
        list = list_files(anno[i])
        total_num = total_num + len(list)

    print('Total Number of Files in Annotations Directories: ' + str(total_num))

#
# We are creating...output files
#       testList.txt file:
#           : This has filename.{jpeg,xml} pairs for the given dataset
#           : This file is used by create_data.sh to create the lmdb data files from dataset
#
#       testImageSet.txt file:
#           : This has a list of all the jpeg files without the extension
#           : This file is used by getImageSize.sh (get_image_size.py), to create the
#           :       name_size_file text file. Changed get_image_size.py to actually
#           :       read testImageList.txt and get the required info. So, the creation of
#           :       this txt file can be removed.
#
#       testImageList.txt file : This has a list of all the jpeg files
#           : This has a list of all the jpeg files with the extension
#           : This is used by ssd_detect binary file to get input image data for running object detection
#
if os.path.exists(testDir):
    shutil.rmtree(testDir)
    print('removed ' + testDir)

if not os.path.exists(testDir):
    os.makedirs(testDir)
if not os.path.exists(testDir + '/Annotations/'):
    os.makedirs(testDir + '/Annotations/')
if not os.path.exists(testDir + '/Data/'):
    os.makedirs(testDir + '/Data/')

dstFile_test = open('%s/testList.txt'%(testDir), 'w')
dstFile_imageList = open('%s/testImageList.txt'%(testDir), 'w')
dstFile_imageSet = open('%s/testImageSet.txt'%(testDir), 'w')
numOfFilesWithNoAnnotations = 0
numOfAnnotatedFilesWithNoImages = 0
numOfUsefulImageFiles = 0

def prepareAndCopyAnnotationAndImageFiles(annoFiles, fileIdx) :

    global numOfFilesWithNoAnnotations
    global numOfAnnotatedFilesWithNoImages
    global numOfUsefulImageFiles

    for annoName in annoFiles:

        #print('annoName in List: ' + annoName)

        annoPath, annoFilename = os.path.split(annoName)
        oldname, annoExt = os.path.splitext(annoFilename)
        #print('path.split: ' + annoPath + ' ' + annoFilename)
        #print('file split name & extension: ' + oldname + ' ' + annoext)

        # replace the rightmost occurrence of 'Annotations' with 'Data'
        #srcImg = 'Data/'.join(annoPath.rsplit('Annotations', 1))
        srcImg = imageDirPrefix.join(annoPath.rsplit('Annotations', 1)) + '/'

        # check if src image exists
        # read a file from the data directory and get the extension
        # Note: assumption is that all files in that directory are the same type
        listSrc = os.listdir(srcImg)
        sfname, sfext = os.path.splitext(listSrc[0])
        srcImgFile = srcImg + oldname + sfext

        # srcImgFile = srcImg + oldname + extension
        merge = testDir
        #dstFile = dstFile_test

        # keep the original name as is, and append the index(fileIdx) to make it unique
        imageId = fileIdx
        dstImg = merge + '/Data/'
        dstImgFile = dstImg + oldname + '-' + str(imageId) + sfext
        dstAnno = merge + '/Annotations/' + oldname + '-' + str(imageId) + annoExt

        fileIdx += 1

        if annoExt == '.xml' :
            annoCount = repairXml(dstAnno, annoName)
        elif annoExt == '.json' :
            annoCount = json2json(annoName, dstAnno, dstImgFile, imageId)
        else :
            raise valueError('Unknown Extension : ' + annoExt)
            numOfFilesWithNoAnnotations += 1
            continue

        #if annoCount > 0 :
        if (annoCount > 0) or (includeAllGt == 'yes'):

            ## For debug purposes
            if fileIdx < 0 :
                print('numOfFilesWithNoAnnotations is: %d'%(numOfFilesWithNoAnnotations))
                print('SrcImg is: ' + srcImg)
                print('srcImgFile full path is: ' + srcImgFile)
                print(' srcAnno is: ' + annoName)
                print('DstImgFile Full path is: ' + dstImgFile)
                print(' DstAnno is: ' + dstAnno)

            if (annoCount == 0) :
                numOfFilesWithNoAnnotations += 1

            # do with only one src/dst file
            if os.path.isfile(srcImgFile):
                shutil.copy2(srcImgFile, dstImgFile)
                #img = cv2.imread(srcImgFile)
                #cv2.imwrite(dstImgFile, img)

                # setup the file name accordingly NOTE: Saving full path name of file
                dstFile_test.write(dstImgFile)
                dstFile_test.write(" ")
                dstFile_test.write(dstAnno)
                dstFile_test.write("\n")

                # NOTE: (km) Creating imageSet.txt file also here
                imgfname, imgfext = os.path.splitext(dstImgFile)
                dstFile_imageSet.write(imgfname)
                dstFile_imageSet.write("\n")

                # NOTE: (km) Creating imageList.txt file also here
                dstFile_imageList.write(dstImgFile)
                dstFile_imageList.write("\n")

                numOfUsefulImageFiles += 1

            else:
                numOfAnnotatedFilesWithNoImages += 1
                # remove the copied annotations in destination folder
                os.remove(dstAnno)

        else:
            numOfFilesWithNoAnnotations += 1

    return fileIdx

def prepareAndCopyUsingAnnoJsonFile(annoJsonFileName, fileIndex, imageDirPrefix) :

    global numOfFilesWithNoAnnotations
    global numOfAnnotatedFilesWithNoImages
    global numOfUsefulImageFiles

    with open(annoJsonFileName) as ajfile:
        ajdata = json.load(ajfile)

    print "========= GT being created based on %s ================"%(annoJsonFileName)
    print "Number of Images: %s"%(str(len(ajdata)))

    imageDirPrefix = "/" + imageDirPrefix.strip("/") + "/"

    # get image data file extension.
    # Assume all the files in that directory have the same extension
    imageExt = '.jpeg'
    dirPath, annoListfile = os.path.split(annoJsonFileName)
    item = ajdata[0]
    tmpFileName = dirPath + '/' + item
    tmpFileName = tmpFileName.replace('/Annotations/', imageDirPrefix)
    tmpFilePath, voidName = os.path.split(tmpFileName)
    listSrc = os.listdir(tmpFilePath + '/')
    voidName, imageExt = os.path.splitext(listSrc[0])
    #print 'image extension : ' + imageExt

    # setup the destination dir
    merge = testDir

    # We have to use the item entry index as the file Index
    fileIndex = 0
    for item in ajdata :

        #print item
        xxpath, xxname = os.path.split(item)
        ifname, annoExt = os.path.splitext(xxname)

        voidName, voidExt = os.path.splitext(item)
        srcImageFile = dirPath + '/' + voidName + imageExt
        srcImageFile = srcImageFile.replace('/Annotations/', imageDirPrefix)
        srcAnnoFile = dirPath + '/' + item

        if not os.path.exists(srcImageFile) or not os.path.exists(srcAnnoFile) :
            print 'Either srcImageFile or SrcAnnoFile do NOT exist'
            print 'srcImageFile: ' + srcImageFile
            print 'srcAnnoFile: ' + srcAnnoFile
            fileIndex += 1
            continue

        # keep the original name as is, and append the index(fileIndex) to make it unique
        imageId = fileIndex
        dstImageFile = merge + '/Data/' + ifname + '-' + str(imageId) + imageExt
        dstAnnoFile = merge + '/Annotations/' + ifname + '-' + str(imageId) + annoExt

        fileIndex += 1

        # prep and copy the files into the destination folders
        if annoExt == '.xml' :
            annoCount = repairXml(dstAnnoFile, srcAnnoFile)
        elif annoExt == '.json' :
            annoCount = json2json(srcAnnoFile, dstAnnoFile, dstImageFile, imageId)
        else :
            raise valueError('Unknown Extension : ' + annoExt)
            numOfFilesWithNoAnnotations += 1
            continue

        if (annoCount > 0) or (includeAllGt == 'yes'):

            ## For debug purposes
            if fileIndex < -1 :
                print('numOfFilesWithNoAnnotations is: %d'%(numOfFilesWithNoAnnotations))
                #print 'srcAnnoFile: ' + srcAnnoFile
                #print 'srcImageFile: ' + srcImageFile
                print 'dstImageFile: ' + dstImageFile
                #print 'dstAnnoFile: ' + dstAnnoFile

            if (annoCount == 0) :
                numOfFilesWithNoAnnotations += 1

            shutil.copy2(srcImageFile, dstImageFile)
            # setup the file name accordingly NOTE: Saving full path name of file
            dstFile_test.write(dstImageFile)
            dstFile_test.write(" ")
            dstFile_test.write(dstAnnoFile)
            dstFile_test.write("\n")

            # NOTE: (km) Creating imageSet.txt file also here
            imgfname, imgfext = os.path.splitext(dstImageFile)
            dstFile_imageSet.write(imgfname)
            dstFile_imageSet.write("\n")

            # NOTE: (km) Creating imageList.txt file also here
            dstFile_imageList.write(dstImageFile)
            dstFile_imageList.write("\n")

            numOfUsefulImageFiles += 1

        else:
            numOfFilesWithNoAnnotations += 1

        ## XXX : hack to get away
        #if numOfUsefulImageFiles > 5 :
        #    break


def prepareAndCopyUsingImageJsonFile(imageJsonFileName, fileIndex, imageDirPrefix) :

    global numOfFilesWithNoAnnotations
    global numOfAnnotatedFilesWithNoImages
    global numOfUsefulImageFiles

    with open(imageJsonFileName) as ijfile:
        ijdata = json.load(ijfile)

    print "========= GT being created based on %s ================"%(imageJsonFileName)
    print "Number of Images: %s"%(str(len(ijdata)))

    ##------------------------------- FIXME HACKs START ---------------------------------------
    #   These are workarounds for now until addressed on the conservator side
    #   NOTE: only previewData works since file extension is jpeg in imageset.json
    #JTM: This is handled in argparse. Following line prefixes and postfixes with '/'
    imageDirPrefix = "/" + imageDirPrefix.strip("/") + "/"
    #imageDirPrefix="/Data/"
    #imageDirPrefix="/tiffData/"
    #imageDirPrefix="/PreviewData/"
    annoDirPrefix="/Annotations/"
    ##-------------------------------- FIXME HACKs END -----------------------------------------

    # get the path from images.json file & grab the annotation file extension.
    # Assume all the files in that directory have the same extension
    dirPath, imgListfile = os.path.split(imageJsonFileName)
    srcAnnoPath = dirPath + '/Annotations/'
    listSrc = os.listdir(srcAnnoPath)
    voidName, annoExt = os.path.splitext(listSrc[0])

    # setup the destination dir
    merge = testDir

    # We have to use the item entry index as the file Index
    fileIndex = 0
    for item in ijdata :
        #print item
        srcImageFile = dirPath + imageDirPrefix + item
        #srcAnnoFile = dirPath + imageDirPrefix + item
        # FIXME HACK ALERT XXX: Until we fix the conservator side. Get the Annotation filenames
        ipath, iname = os.path.split(item)
        ifname, iext = os.path.splitext(iname)
        srcAnnoFile = dirPath + annoDirPrefix + ifname + annoExt

        if not os.path.exists(srcImageFile) or not os.path.exists(srcAnnoFile) :
            print 'Either srcImageFile or SrcAnnoFile do NOT exist'
            print 'srcImageFile: ' + srcImageFile
            print 'srcAnnoFile: ' + srcAnnoFile
            fileIndex += 1
            continue

        # keep the original name as is, and append the index(fileIndex) to make it unique
        imageId = fileIndex
        dstImageFile = merge + '/Data/' + ifname + '-' + str(imageId) + iext
        dstAnnoFile = merge + '/Annotations/' + ifname + '-' + str(imageId) + annoExt

        fileIndex += 1

        # prep and copy the files into the destination folders
        if annoExt == '.xml' :
            annoCount = repairXml(dstAnnoFile, srcAnnoFile)
        elif annoExt == '.json' :
            annoCount = json2json(srcAnnoFile, dstAnnoFile, dstImageFile, imageId)
        else :
            raise valueError('Unknown Extension : ' + annoExt)
            numOfFilesWithNoAnnotations += 1
            continue

        #if annoCount > 0 :
        if (annoCount > 0) or (includeAllGt == 'yes'):

            ## For debug purposes
            if fileIndex < -1 :
                print('numOfFilesWithNoAnnotations is: %d'%(numOfFilesWithNoAnnotations))
                #print 'srcAnnoFile: ' + srcAnnoFile
                #print 'srcImageFile: ' + srcImageFile
                print 'dstImageFile: ' + dstImageFile
                #print 'dstAnnoFile: ' + dstAnnoFile

            if (annoCount == 0) :
                numOfFilesWithNoAnnotations += 1

            shutil.copy2(srcImageFile, dstImageFile)
            # setup the file name accordingly NOTE: Saving full path name of file
            dstFile_test.write(dstImageFile)
            dstFile_test.write(" ")
            dstFile_test.write(dstAnnoFile)
            dstFile_test.write("\n")

            # NOTE: (km) Creating imageSet.txt file also here
            imgfname, imgfext = os.path.splitext(dstImageFile)
            dstFile_imageSet.write(imgfname)
            dstFile_imageSet.write("\n")

            # NOTE: (km) Creating imageList.txt file also here
            dstFile_imageList.write(dstImageFile)
            dstFile_imageList.write("\n")

            numOfUsefulImageFiles += 1

        else:
            numOfFilesWithNoAnnotations += 1

        ## XXX : hack to get away
        #if numOfUsefulImageFiles > 5 :
        #    break

fileIndex = 0
if not os.path.exists(annoJsonFileName):
    print "============ GT being created based on Annotation Directory =============="
    gatherAnnotationDirs(srcDir)
    for i in range(0,len(anno)):
        annoFileList = list_files(anno[i])
        fileIndex = prepareAndCopyAnnotationAndImageFiles(annoFileList, fileIndex)
else :
    # use the passed json file
    prepareAndCopyUsingAnnoJsonFile(annoJsonFileName, fileIndex, imageDirPrefix)

#dstFile.close()
dstFile_test.close()
dstFile_imageSet.close()
dstFile_imageList.close()

print('Number of Files with No Annotations are: ' + str(numOfFilesWithNoAnnotations))
print('Number of Annotated Files with no Images are: ' + str(numOfAnnotatedFilesWithNoImages))
print('Number of Useful Files are: ' + str(numOfUsefulImageFiles))
