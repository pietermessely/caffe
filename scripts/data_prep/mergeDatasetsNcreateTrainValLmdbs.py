'''
Q2'2018:

    - Added support to selectively add gaussian blur at a RoI (based on bounding box).
        This is controlled by the input argument : gaussianBlurPercentage
    - Support images for Hard-Negative Mining implicitly.
         Modified to check for Data directory(images in directory).
         If no corresponding 'Annotations" directory/file, will be treated as
         data for 'Hard-Negative' mining.

Q1'2018: Some of the code is adapted and modified from existing files
@author: kedar m

mergeDatasetsNcreateTrainValLmdbs.py
------------------------------------

Merge different dataset sources into one large dataset for use with training
The differences between datasets that are accounted for are:
1. Annotations: can be in 'xml' or 'json' formats
2. Filenames & extensions:   conservator format or non-conservator format
3. image extensions: 'jpeg' from conservator, 'jpg' from coco-datasets
This routine takes input argument to get full-path to the root dataset dir. This
    can have multiple datasets in sub-directories that contain the corresponding
    Annotation and Data dirs. The initial version only checks for Annotation dirs
    two level deep. An example structure that is known to work is:

                                   root-ds
                                      |
                -----------------------------------------------------------
                /       |           |       \                    \
            ds-foo-1   ds-foo-2  Annotations Data               merged
        ---------       --------                     -----------------------
            /   \       /      \                   /     /       |        \
  Annotations Data Annotations Data         Annotations Data   train      val
                                                              --------  --------
                                                                / | \    / | \
                                                              A   D lmdb A D lmdb
It figures out the rest i.e.,
    1. creates a 'merged' directory under "root-ds" with
        Annotation, Data sub directories
    2. converts & massages annotations from foo-1 & foo-2 to json format,
            and places them into the merged/Annotations dir, along with
        a. assignment of "image-ids" and augments the filename with these unique IDs
        b. updates the dst image filenames also and copies into merged/Data dirs
        c. rewrites relevant fields in the annotation files accordingly to reflect these
        d. copies image files from foo-1/Data and foo-2/Data into merged/Data dirs
    4. creates relevant fooList.txt files including name_size files
    5. Splits the src datasets into train and val datasets (20% split ratio)
    6. calls "annotate" python script with set arguments to generate LMDBs
    7. roots-ds/merged/train/lmdb & root-ds/merged/val/lmdb are generated that can
        be used in training a CNN

The initial version of this script takes these mandatory args in order:
    1. source dataset dir (root-ds above)
    2. categoryIds.json file (labelmap)
    3. labelvector.json file (labelvector)
    4. gaussianBlurPercentage [0..99]
    5. categoryId2check [0..90]

Optional args:
    1. prefixDir           : Replace occurences of 'Data' with prefixDir. As of the time of this
                             writing, Conservator exports 8bit images in "PreviewData".
    2. remap_labels        : Json file specifying lables to remap.
                             Ex: { src_label: dst_label }
    3. bbox_area_threshold : minimum area of bbox for inclusion
    4. val_split           : Percent of images to use for validation
    5. keep_all_annos      : Don't discard annotations that aren't in your labelmap or labelvector. Without this flag, they will be discarded.
    6. discard_classes     : Comma separated list of labelmap ids to discard. Ex: 3,7,84

Hard-Negative Mining:
    This script supports inclusion of images implicitly for hard-negtive mining.
    A 'Data' directory with no corresponding 'Annotations' directory will result
    in using the images under 'Data' directory as images with NO annotations.
    It keeps a tab of hard-negative files by creating an annotation file, with
    empty annotations and fills the image details. It adds 'hard_negative' key
    in the image details, to weed out non hard_negative images with no annotations
    later as part of categoryId based filtering. So, first time you run this script
    with hard-negative images, make sure you remove the created 'Annotations'
    folder in this directory, with any prior runs or remove always.

Misc: keep a count of each annotation in the dataset
    This script also counts the number of 'small/medium/large' objects in the datasets,
    for each label type. These are dumped into a file (xxxx_sizes_list.txt)

Gaussian Blur on RoI: add blur to gaussianBlurPercentage of imgaes in 'train' datasets.
    This is in the annotated area(bbox) or RoI, using a kernel of 5 and sigma of 2.
    If you do this, don't use caffe's 'gauss_blur' data augmentation - which actually
    blurs the entire image.

CategoryId2check: (Re)moves images with NO annotations for categoryId2check provided as input to
    a separate directory (rem) under 'merged' dir. These are not included in the LMDBs.
    If the categoryId2check input is '0', then this is disabled i.e., no check is done.

Example run command:
    python mergeDatasetsNcreateTrainValLmdbs.py /path/to/dataset /path/to/coco_catids.json /path/to/labelvector.json 0 0 --val_split 13 --discard_classes 81,82,83,87 --remap_labels /path/to/mapping.json
'''

import os
import xml.etree.cElementTree as ET
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import json
from random import shuffle
import shutil
import argparse
import cv2

from PIL import Image
import sys
import subprocess
import blurRoi as br
import labelmap_utils
from collections import OrderedDict


caffeRoot = os.environ.get("CAFFE_ROOT")
if (caffeRoot is None) :
    print "Need to setup CAFFE_ROOT. Cannot Proceed Further"
    raise Exception ("CAFFE_ROOT : Not Set")
    sys.exit()

pythonPath = os.environ.get("PYTHONPATH")
if (pythonPath is None) :
    print 'Need to setup enviromental variable PYTHONPATH to proceed. Need caffe proto'
    raise Exception ('Need to setup enviromental variable PYTHONPATH to proceed')
else :
    print 'PYTHONPATH : {}'.format(pythonPath)


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataSetDir", type=str,
                    help="Source Directory of Data Set to be merged : FULL PATH")
parser.add_argument("cocoCatIdsFileName", type=str,
                    help="Name of file containing the source annotation IDs, used for creating the label map")
parser.add_argument("labelFile", type=str,
                    help="Name of labelvector. The labelvector is a vector of labels with no holes.")
parser.add_argument("gaussianBlurPercentage", type=int, choices=range(0,100),
                    help="% of training images on which Gaussian Blur will be applied")
parser.add_argument("catId2check", type=int, choices=range(0,90),
                    help="Images without any annotations of this categoryId will be excluded from training & validation datasets. 0: None, 1-90: specific category")
parser.add_argument("--prefix", type=str, help="Change the directory in which images are found (typically: 'Data') to something else")
parser.add_argument("--remap_labels", type=str, help="JSON file specifying label renaming of annotations to catids as { src_label: dst_label }")
parser.add_argument("--bbox_area_threshold", type=int, help="minimum area of bbox for inclusion")
parser.add_argument("--bbox_width_threshold", type=int, help="minimum width of bbox for inclusion")
parser.add_argument("--bbox_height_threshold", type=int, help="minimum height of bbox for inclusion")
parser.add_argument("--val_split", type=int, help="percent of images to use for validation", choices=range(0, 100), default=20)
parser.add_argument("--keep_all_annos", action="store_false", help="If an annotation is not in your labelvector or labelmap, keep it anyways.")
parser.add_argument("--discard_classes", type=str, help="Comma separated list of labelmap ids. Classes not in this list will not be included in merged annotations. (ex: 1,5,71,88)")
parser.add_argument("--resize", type=str, help="WxH, dimensions to resize all images to. It will retain the aspect ratio of the original image, and fill extra with black bars.")
parser.add_argument("--skip_create_lmdb", type=str2bool, default=False, help="Skip creating training LMDBs (default: False)")
parser.add_argument("--force_copy", type=str2bool, default=False, help="Force script to make copies of the images instead of symlinks")
parser.add_argument("--output_dir", type=str, help="Full path to directory to output merged dataset to, if not specified default is dataSetDir/merged")
parser.add_argument("--blur_removed_annos",action='store_true',help="If argument is used blurs the area within annotations that are removed for size")
parser.add_argument("--hideNseek_classes", type=str, default=None, help="add hideNSeek augmentations for the comma separated list of category Ids. (ex: 1,2)")

args = parser.parse_args()
dataSrcDir = args.dataSetDir
cocoCatIdsFileName = args.cocoCatIdsFileName
labelFile = args.labelFile
prefixDir = args.prefix or "Data"

hideNseek_classes = args.hideNseek_classes
if hideNseek_classes : 
    #hideNseek_class_list = set(args.hideNseek_classes.split(','))
    hideNseek_class_list = (args.hideNseek_classes.split(','))
else : 
    #hideNseek_class_list = set([])
    hideNseek_class_list = []
print '=== hideNseek_class_list is {} ===='.format(hideNseek_class_list)

discard_classes = args.discard_classes
if discard_classes:
    discard_class_list = set(args.discard_classes.split(','))
else:
    discard_class_list = set([])
check_allowed_classes = args.keep_all_annos
resize_images = False
resize_width, resize_height = [None, None]
if args.resize:
    resize_images = True
    resize_width, resize_height = [int(dim) for dim in args.resize.split('x')]
skip_create_lmdb = args.skip_create_lmdb

# Load label remap
remap_label_to_catid_name = {}
if args.remap_labels:
    print "Loading label remap file: %s" % args.remap_labels
    f = open(args.remap_labels, 'r')
    remap_label_to_catid_name = json.load(f)

# 0: None, 1: person, 2: bicycle, 3: car ....
#hideNseekAugmentation = args.hideNseek
gbPercentage = int(args.gaussianBlurPercentage)
catId2check = int(args.catId2check)
bbox_area_threshold = args.bbox_area_threshold or 1
width_threshold = 1
height_threshold = 1

print 'CAUTION: bbox threshold value: %d '%(bbox_area_threshold)
bbox_threshold_based_exclusions = 0
width_threshold_based_exclusions = 0
height_threshold_based_exclusions = 0

## =================================================================================

mergeDS = 'merged'
baseDir = '%s'%(dataSrcDir)
mergeDir = args.output_dir or '%s/%s'%(baseDir,mergeDS)

trainDir = '%s/train'%(mergeDir)
taDir = '%s/Annotations/'%(trainDir)
tdDir = '%s/%s/'%(trainDir, prefixDir)

valDir = '%s/val'%(mergeDir)
vaDir = '%s/Annotations/'%(valDir)
vdDir = '%s/%s/'%(valDir, prefixDir)

valSplit = args.val_split

# setup required output directory names
dstAnnoDir = '%s/Annotations/'%(mergeDir)
dstImageDir = '%s/%s/'% (mergeDir, prefixDir)

print ("prefixDir: " + prefixDir)
print ("dstAnnoDir: " + dstAnnoDir )
print ("dstImageDir: " + dstImageDir )
print ("mergeDir: " + mergeDir )

# remove existing dst directory, then create the requried dirs & sub-dirs
if os.path.exists(mergeDir):
    shutil.rmtree(mergeDir)

# Create mapping of labels to ids
name_to_coco_catid = labelmap_utils.get_name_to_coco_catid(cocoCatIdsFileName)
name_to_labelvector_id = labelmap_utils.get_name_to_labelvector_id(labelFile)
labelvector = labelmap_utils.get_labelvector(labelFile)
allowed_coco_id_list = set([str(name_to_coco_catid.get(label_i, 0)) for label_i in labelvector]) - discard_class_list
print "accepted classes: "
print sorted(allowed_coco_id_list)

# Notify user about remaping
for src_label, dst_label in remap_label_to_catid_name.items():
    if not dst_label in name_to_coco_catid:
        print "WARNING: Cannot remap label %s because destination label '%s' is not in your coco catids"
        continue
    dst_catid = name_to_coco_catid[dst_label]
    if dst_label in name_to_labelvector_id:
        dst_labelvector_id = str(name_to_labelvector_id[dst_label])
        print "REMAP: %s -> %s (catid: %s, labelvectorid: %s)" % (src_label, dst_label, dst_catid, dst_labelvector_id)
    else:
        print "REMAP: %s -> %s (catid: %s)  WARNING: This label is not in your labelvector" % (src_label, dst_label, dst_catid)

# This function gets called to remap the ids of all the annotations to the labelvector index
def getCocoID(label):
    # Do remapping
    if label in remap_label_to_catid_name:
        label = remap_label_to_catid_name[label]
    if label not in name_to_coco_catid:
        #print label + ' Not Found!!'
        return -1
    else:
        return str(name_to_coco_catid[label])

annotations_with_no_catid = set()
files_with_labels_not_in_catids = set()
def xml2json(dstFile, xmlFile, imgFile, image_id):
    global bbox_threshold_based_exclusions
    global width_threshold_based_exclusions
    global height_threshold_based_exclusions

    noLabels = []
    anno_id = 1

    tree = ET.parse(xmlFile)
    root = tree.getroot()

    size = root.find('size')
    imgWidth = int(float(size.find('width').text))
    imgHeight = int(float(size.find('height').text))
    img = {'file_name': '{}'.format(imgFile), 'height': imgHeight, 'id': image_id, 'width': imgWidth}

    anns = []
    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        if bndbox is None:
            continue

        xmin = float(bndbox.find('xmin').text)
        xmin = int(xmin)
        xmax = float(bndbox.find('xmax').text)
        xmax = int(xmax)
        ymin = float(bndbox.find('ymin').text)
        ymin = int(ymin)
        ymax = float(bndbox.find('ymax').text)
        ymax = int(ymax)

        x = xmin - 1
        y = ymin - 1

        # clamp values to image bounds
        x = max(0, x)
        y = max(0, y)
        xmax = min(imgWidth - 1, xmax)
        ymax = min(imgHeight - 1, ymax)

        width = (xmax - xmin) + 1
        height = (ymax - ymin) + 1

	if width < width_threshold :
	    width_threshold_based_exclusions += 1
	    continue

	if height < height_threshold :
	    height_threshold_based_exclusions += 1
	    continue

        area = width * height
        bbox = [x, y, width, height]

        if area < bbox_area_threshold:
            bbox_threshold_based_exclusions += 1
            continue

        category_str = object.find('name').text.lower()
        category_id = getCocoID(category_str)
        if check_allowed_classes:
            if category_id not in allowed_coco_id_list:
                #print "Coco category id: {} is not allowed, and will be ignored.".format(category_id)
                continue
        if (category_id < 0):
            annotations_with_no_catid.add(category_str)
            files_with_labels_not_in_catids.add(xmlFile)
            continue

        segmentation=[x, y, x, ymax, xmax, ymax, xmax, y]

        anns.append({'area': area, 'bbox': bbox, 'category_id': category_id, 'id': anno_id,  'image_id': image_id,  'iscrowd': 0, 'segmentation': [segmentation]})

        anno_id = anno_id + 1

    data = {'annotation': anns, 'image': img}

    with open(dstFile, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, separators=(',', ':'))


# ====================================================
# gather file names into a list
# ====================================================
def list_files(path):
    # returns a list of names (with extension, and full path) of all files
    files = []
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        files.extend(map(lambda f : os.path.join(root, f), filenames))
    #print files[0]
    return files

# ====================================================
# count the number of files in the provided list of dirs
# ====================================================
def countNumOfFiles(dirlist, dirName):
    total_num = 0
    for i in range(0, len(dirlist)):
        fl = list_files(dirlist[i])
        total_num = total_num + len(fl)

    print('Total Number of Files in {} dirs: {}'.format(dirName, total_num))

# ====================================================
# Gather list of directories named 'dirName'.
# Typically, Each annotation should have a corresponding image file.
# By convention, image and annotation
# files have the same name, only the extension is different.  ie image0.jpg and image0.[xml,json]
# searching for a directory named dirName - only two levels deep for now
# ====================================================

def getDirsByName(rootdir, dirName):
    odirs = []
    for name in os.listdir(rootdir):
        srcDir2 = rootdir + '/' + name + '/'
        #if (os.path.isdir(srcDir2)) & (name == 'Annotations'):
        #if name == 'Annotations':
        if name == dirName:
            odirs.append(srcDir2)
        elif os.path.isdir(srcDir2):
            for name2 in os.listdir(srcDir2):
                srcDir3 = srcDir2 + name2 + '/'
                if name2 == dirName:
                    odirs.append(srcDir3)
                else:
                    #print('ignoring: ' + srcDir3)
                    pass
        else:
            pass
    #countNumOfAnnFiles(adirs)
    countNumOfFiles(odirs, dirName)
    return odirs

# ====================================================
# remove annotations from file
# ====================================================
def removeAnnotations(srcFile, dstFile):
    #print ('removeAnnotations: '+ srcFile + '\n dstFile: ' + dstFile)
    # setup input file and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    image = idata["image"]
    #anns = idata["annotation"]
    idata["annotation"] = []

    ofile = open(dstFile, 'w')
    json.dump(idata, ofile, indent=4, sort_keys=True, separators=(',',':'))

# ====================================================
# create the new json file from old one and update 'file_name'
#       field of the 'image' node in dst file to passed fileName and
#       update the 'image_id' field in each annotation
# ====================================================
def json2json(srcFile, dstFile, fileName, imageId):
    global bbox_threshold_based_exclusions
    global width_threshold_based_exclusions
    global height_threshold_based_exclusions

    #print ('json2json srcFile: '+ srcFile + 'dstFile: ' + dstFile + 'fileName: ' + fileName)
    #print ('imageId: %s'%(imageId))
    # setup input file and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    image = idata["image"]
    # update image data
    image["file_name"] = fileName
    image["id"] = imageId
    removed_anno_bbox = []

    #change annotation data
    anns = idata["annotation"]
    numOfAnns = len(anns)
    i = 0
    while i < numOfAnns:
        if check_allowed_classes:
            if str(anns[i]["category_id"]) not in allowed_coco_id_list:
                #print "Coco category id: {} is not allowed, and will be ignored.".format(str(anns[i]["category_id"]))
                del anns[i]
                numOfAnns = numOfAnns-1
                continue
	    #print 'i is %d, numOfAnns: %d '%(i, len(anns))
        anns[i]["image_id"] = imageId
        anns[i]["id"] = i

        # Hack to convert mar5_drone_20K dataset from google-drive to follow the prior
        # synthetic train-tag-drone-jan18 dataset i.e., use "89" instead of "79" for drone
        #if (anns[i]["category_id"] == 79) :
        #   anns[i]["category_id"] = 89
        #print 'new imageid: %s'%(anns[i]["image_id"])

	# remove annotations with area < the requested threshold
        # also: height or width less than threshold
	area = int(anns[i]["area"])
        width = int(anns[i]["bbox"][2])
        height = int(anns[i]["bbox"][3])
	if area < bbox_area_threshold :
	    bbox_threshold_based_exclusions += 1
	    removed_anno_bbox.append(anns[i]["bbox"])
	    #anns.remove(anns[i])
	    del anns[i]
	    numOfAnns = numOfAnns-1
        elif width < width_threshold :
	    width_threshold_based_exclusions += 1
	    removed_anno_bbox.append(anns[i]["bbox"])
	    del anns[i]
	    numOfAnns = numOfAnns-1
        elif height < height_threshold :
	    height_threshold_based_exclusions += 1
	    removed_anno_bbox.append(anns[i]["bbox"])
	    del anns[i]
	    numOfAnns = numOfAnns-1
        else:
	        i = i+1
    ofile = open(dstFile, 'w')
    json.dump(idata, ofile, indent=4, sort_keys=True, separators=(',',':'))
    return(removed_anno_bbox)

# ===================================================================
# Logic below follows these steps :
#       1. fix file-names to follow conservator format i.e., a-b-c-d-e.[json,jpeg]
#       2. fix image file extensions to be 'jpeg' for all
#       3. augment file-name with unique sequence number i.e., imageID
#       4. convert each xml to its corresponding json
#       5. rewrite file_name & image_id in existing json annotation files
#       6. create the new Annotations and images files in the dst folders
#       7. split into 'train' & 'val' sets and create LMDBs
#
#       #. generates output list files along side
#          xxxList.txt file:
#               : This has filename.{jpeg,xml} pairs for the given dataset
#               : This file is used by create_data.sh to create the lmdb data files from dataset
#          xxxImageList.txt file :
#               : This has a list of all the jpeg files with the extension
#               : This is used by ssd_detect binary file to get input image data for running object detection
#               : This is used by get_image_size.py to generate the name_size.file
#          xxx_name_size.txt file :
#               : This has a list of image_ids with their sizes
# ===================================================================

newIdx = 1
numOfFilesWithNoAnnotations = 0
numOfAnnotatedFilesWithNoImages = 0
numOfUsefulImageFiles = 0

## =======================================================
#       The following goes through list of files in prefixDir
#       dir and sets up Annotation+Image file paris, with the
#       final annotations in 'json' format. It converts
#       any 'xml' annotations into 'json' format in the
#       process. If there are no annotations file/dir, it
#       creates a dummy one assuming, image is for hard-negative
#       mining. It returns the 'next' available index
#       for use in the file names as sequence number.
## ======================================================
def setupAnnoFilePairFromDataDirs(srcDir, fileSeqNum, ailFile, ilFile, nsFile, numOrphanAnnoFiles, numImageFiles):
    ## XXX: Moving away from this. No hard-coded vents
    #conservator orig filename: video-D6YuLc44qf4k62YWn-frame-000140-sQcRbCiwHfr6joJXY
    #origConservatorFilenameSplitLength=5
    imageFileExt = 'NONE'
    annoFileExt = '.json'
    listDfiles = list_files(srcDir)
    #print("in:dir: %s, numOfFiles: %s, in:fileSeqNum %d \n"%(srcDir, len(listDfiles), fileSeqNum))
    #print("in:numOrphanAnnoFiles: %d, numImageFiles: %d\n"%(numOrphanAnnoFiles, numImageFiles))
    # replace rightmost occurrence of prefixDir with 'Annotations' for corresponding Anno dir
    srcAnnoDir = 'Annotations'.join(srcDir.rsplit(prefixDir, 1))
    #print('srcAnnoDir is :' + srcAnnoDir)
    # read a file from the Anno directory and get the extension
    # Note: assumption is that all files in that directory are the same type
    anno_filenames = []
    anno_basenames = []
    anno_names_we = []
    if os.path.isdir(srcAnnoDir):
        anno_filenames = list_files(srcAnnoDir)
        anno_basenames = map(os.path.basename, anno_filenames)
        anno_names_we = map(lambda b : os.path.splitext(b)[0], anno_basenames)
    else:
        print '=======: creating Dir: ' + srcAnnoDir
        annoFileExt = '.json'
        os.makedirs(srcAnnoDir)
    # This maps the annotation filename without extension to the full path of the annotation file
    name_anno_map = {}
    if len(anno_filenames) > 0:
        # Grab extension of first annotation
        _, annoFileExt = os.path.splitext(anno_basenames[0])
        # Create mapping of {name_we : filename}
        name_anno_map = dict(zip(anno_names_we, anno_filenames))

    print 'annoFile ext: ' + annoFileExt

    for srcImageFile in listDfiles:
        #print('srcImageFile in List: ' + srcImageFile + '\n')
        dataPath, dataFilename = os.path.split(srcImageFile)
        oldname, dext = os.path.splitext(dataFilename)
        #print('path.split: ' + dataPath + ' ' + dataFilename)
        #print('file split name & extension: ' + oldname + ' ' + extension + '\n')
        if (imageFileExt == 'NONE'):
            imageFileExt = dext

        # initialize
        remove_bboxes = []

        newName = oldname
        # get the image size
        im = Image.open(srcImageFile)
        iwidth, iheight = im.size
        #img = cv2.imread(srcImageFile)
        #iheight, iwidth = img.shape[:2]
        # check if srcAnnoFile exists and if none, create one
        #srcAnnoFile = srcAnnoDir + newName + annoFileExt
        # Directly look up the annotation file
        if oldname in name_anno_map:
            srcAnnoFile = name_anno_map[oldname]
        else:
            # Construct a fake annotation file.
            srcAnnoFile = srcAnnoDir + oldname + annoFileExt
            #print 'Created Annotation File: ' + srcAnnoFile
            print 'Image File with No Annotation File: ' + srcImageFile

            #print '====== 2: creating File : ' + srcAnnoFile
            image = {'file_name': '{}'.format(srcImageFile), 'height': iheight, 'id': fileSeqNum, 'width': iwidth, 'hard_negative': 'True'}
            anns = []
            idata = {'annotation': anns, 'image': image}
            ifile = open(srcAnnoFile, 'w')
            json.dump(idata, ifile, indent=4, sort_keys=True, separators=(',', ':'))
            ifile.close()
        #print('srcAnnoFile: ' + srcAnnoFile )
        if annoFileExt == '.json':
            # prepend full path to the filename
            dstAnnoFile = dstAnnoDir + newName + '-' + str(fileSeqNum) + '.json'
            #print ('dstAnnoFile: ' + dstAnnoFile)
            dstImageFile = dstImageDir + newName + '-' + str(fileSeqNum) +  imageFileExt
            #print ('dstImageFile: ' + dstImageFile)
            # update with required 'image' and 'annotation' fields
            remove_bboxes = json2json(srcAnnoFile, dstAnnoFile, dstImageFile, fileSeqNum)
        elif annoFileExt == '.xml':
            # need to convert to json file, assume it is an 'xml' file with the right filename format
            #print('xml file')
            dstAnnoFile = dstAnnoDir + oldname + '-' + str(fileSeqNum) + '.json'
            dstImageFile = dstImageDir + oldname + '-' + str(fileSeqNum) + imageFileExt
            #xml2json(dstFile, xmlFile, imgFile, image_id)
            xml2json(dstAnnoFile, srcAnnoFile, dstImageFile, fileSeqNum)
        else:
            continue

        # copy srcImage into the dst Image dir and also update the required list-name files
        if args.blur_removed_annos and len(remove_bboxes) > 0:

            #if annoFileExt == '.xml':
            #    raise Exception('Bluring not yet enabled for .xml files')

            #if len(remove_bboxes) == 0:
            #    shutil.copy2(srcImageFile, dstImageFile)
            #else:
            kernelSize = 15
            sigma = 6
            contours = [[np.zeros((4, 2), dtype=int)]] * len(remove_bboxes)
            for i, bbox in enumerate(remove_bboxes):
                x, y, w, h = bbox
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                contours[i] = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
            img = br.BlurContours(img, contours, kernelSize, sigma)
            cv2.imwrite(dstImageFile, img)
        else:
            if args.force_copy == True:
              shutil.copy2(srcImageFile, dstImageFile)
            else:
              os.symlink(srcImageFile, dstImageFile)

        # update annoImageList file
        ailFile.write(dstImageFile)
        ailFile.write(" ")
        ailFile.write(dstAnnoFile)
        ailFile.write("\n")

        # update imageList file
        ilFile.write(dstImageFile)
        ilFile.write("\n")

        # update name_size file
        nsFile.write(str(fileSeqNum) + " " + str(iheight) + " " + str(iwidth))
        nsFile.write("\n")

        numImageFiles += 1
        fileSeqNum += 1
    print("dir: %s, num_of_image_files: %d, fileSeqNum: %s \n"%(srcDir, numImageFiles, str(fileSeqNum)))
    return fileSeqNum

## =======================================================
#       The following goes through list of "Annotations" dirs
#       and sets up Annotation+Image file paris, with the
#       final annotations in 'json' format. It converts
#       any 'xml' annotations into 'json' format in the
#       process. It returns the 'next' available index
#       for use in the file names as sequence number.
## ======================================================
def setupDataFilePairFromAnnotationDirs(srcDir, fileSeqNum, ailFile, ilFile, nsFile, numOrphanAnnoFiles, numImageFiles):
    listAfiles = list_files(srcDir)

    #print("in:dir: %s, numOfFiles: %s, in:fileSeqNum %d \n"%(srcDir, len(listAfiles), fileSeqNum))
    #print("in:numOrphanAnnoFiles: %d, numImageFiles: %d\n"%(numOrphanAnnoFiles, numImageFiles))
    imageFileExt = 'NONE'

    # replace rightmost occurrence of 'Annotations' with prefixDir for corresponding image dir
    srcImageDir = prefixDir.join(srcDir.rsplit('Annotations', 1))
    #print('srcImageDir is :' + srcImageDir)

    # read a file from the images directory and get the extension
    # Note: assumption is that all files in that directory are the same type
    ## TODO: loop until we read a valid fle type i.e., jpeg, jpg, tiff, png
    if (imageFileExt == 'NONE'):
        listSrc = os.listdir(srcImageDir)
        sfname, sfext = os.path.splitext(listSrc[0])
        imageFileExt = sfext
        #print 'imageFile ext: ' + imageFileExt

    for srcAnnoFile in listAfiles:
        #print('srcAnnoFile in List: ' + srcAnnoFile + '\n')
        annoPath, annoFilename = os.path.split(srcAnnoFile)
        oldname, extension = os.path.splitext(annoFilename)
        #print('path.split: ' + annoPath + ' ' + annoFilename)
        #print('file split name & extension: ' + oldname + ' ' + extension + '\n')
        if extension == '.json':
            newName = oldname
            # prepend full path to the filename
            dstAnnoFile = dstAnnoDir + newName + '-' + str(fileSeqNum) + '.json'
            #print ('dstAnnoFile: ' + dstAnnoFile)
            dstImageFile = dstImageDir + newName + '-' + str(fileSeqNum) +  imageFileExt
            #print ('dstImageFile: ' + dstImageFile)
            # create new json file and update required 'image' and 'annotation' fields
            json2json(srcAnnoFile, dstAnnoFile, dstImageFile, fileSeqNum)
        else:
            # need to convert to json file, assume it is an 'xml' file with the right filename format
            dstAnnoFile = dstAnnoDir + oldname + '-' + str(fileSeqNum) + '.json'
            dstImageFile = dstImageDir + oldname + '-' + str(fileSeqNum) + imageFileExt
            #xml2json(dstFile, xmlFile, imgFile, image_id)
            xml2json(dstAnnoFile, srcAnnoFile, dstImageFile, fileSeqNum)

        srcImageFile = srcImageDir + oldname + imageFileExt
        # copy srcImage into the dst Image dir and also update the required list-name files
        if os.path.isfile(srcImageFile):
            # get the image size
            img = cv2.imread(srcImageFile)
            height, width = img.shape[:2]
            # use 'os' copy command?
            #cv2.imwrite(dstImageFile, img)
            if args.force_copy == True:
              shutil.copy2(srcImageFile, dstImageFile)
            else:
              os.symlink(srcImageFile, dstImageFile)

            # update annoImageList file
            ailFile.write(dstImageFile)
            ailFile.write(" ")
            ailFile.write(dstAnnoFile)
            ailFile.write("\n")
            # update imageList file
            ilFile.write(dstImageFile)
            ilFile.write("\n")
            # update name_size file
            nsFile.write(str(fileSeqNum) + " " + str(height) + " " + str(width))
            nsFile.write("\n")

            numImageFiles += 1
        else:
            ## remove the annotation files for now & adjust newIdx to accommodate split ratio
            numOrphanAnnoFiles += 1
            fileSeqNum -= 1
            #print('Deleting Annotation File with no Image File: %s'%(dstAnnoFile))
            #shutil.rmtree(dstAnnoFile)
            os.remove(dstAnnoFile)

        fileSeqNum += 1

    print("dir: %s, num_of_image_files: %d fileSeqNum: %s \n"%(srcDir, numImageFiles, str(fileSeqNum)))
    return fileSeqNum

# =======================================================
# for each anno or data dir, call setup routine to massage the annotations
#       and find the corresponding image file in the data dir or viceversa.
# =======================================================
# gather the list of Annotations and prefixDir dirs
dirName = 'Annotations'
annDirs = getDirsByName(baseDir, dirName)
#print('Number of %s DIRs found: %s'%(dirName, len(annDirs)))

dirName = prefixDir
dataDirs = getDirsByName(baseDir, dirName)
#print('Number of %s DIRs found: %s'%(dirName, len(dataDirs)))

# create output dirs
os.makedirs(mergeDir)
os.makedirs(dstAnnoDir)
os.makedirs(dstImageDir)

annoImageListFile = open('%s/anno_image_list.txt'%(mergeDir), 'w')
imageListFile = open('%s/image_list.txt'%(mergeDir), 'w')
nameSizeFile = open('%s/name_size_list.txt'%(mergeDir), 'w')

## based on input argument, either setup based on data dirs or anno dirs
setupBasedOnDataDir = False
setupBasedOnDataDir = True

if setupBasedOnDataDir == True:
    for idir in range(0,len(dataDirs)):
        dataSrcDir = dataDirs[idir]
        #print("calling: dataSrcDir: %s, newIdx: %d \n"%(dataSrcDir, newIdx))
        newIdx = setupAnnoFilePairFromDataDirs(dataSrcDir, newIdx, annoImageListFile, imageListFile, nameSizeFile, numOfAnnotatedFilesWithNoImages, numOfUsefulImageFiles)
        #print("return: dataSrcDir: %s, newIdx: %d \n"%(dataSrcDir, newIdx))
else:
    for idir in range(0, len(annDirs)):
        annoSrcDir = annDirs[idir]
        #print("calling: annoSrcDir: %s, newIdx: %d \n"%(annoSrcDir, newIdx))
        newIdx = setupDataFilePairFromAnnotationDirs(annoSrcDir, newIdx, annoImageListFile, imageListFile, nameSizeFile, numOfAnnotatedFilesWithNoImages, numOfUsefulImageFiles)
        #print("return: annoSrcDir: %s, newIdx: %d \n"%(annoSrcDir, newIdx))
# Notify user about suspicious labels
if len(annotations_with_no_catid) > 0:
    print "NOTE: The following labels were found in annotations but do not have a corresponding catid:"
    for label in annotations_with_no_catid:
        print "  " + label
    print "The %d files in which these labels can be found are listed in /tmp/files_with_labels_not_in_catids.txt" % len(files_with_labels_not_in_catids)
    print ""
    f = open("%s/files_with_labels_not_in_catids.txt" % mergeDir, 'w')
    f.writelines("\n".join(files_with_labels_not_in_catids))

annoImageListFile.close()
imageListFile.close()
nameSizeFile.close()

#print('num_annoFiles_with_no_image_files: %s, num_of_useful_img_files: %s\n'%(numOfAnnotatedFilesWithNoImages, numOfUsefulImageFiles))

print ' Number of bbox threshold based annotations removed are : {} '.format(bbox_threshold_based_exclusions)
print ' Number of width threshold based annotations removed are : {} '.format(width_threshold_based_exclusions)
print ' Number of height threshold based annotations removed are : {} '.format(height_threshold_based_exclusions)

# =======================================================
# function to generate indices for use by train/val split
# =======================================================
def create_test_idx(num_total_samples, percentage):
    num_test_samples = int(num_total_samples*percentage*0.01)
    if (num_test_samples < 1):
        num_test_samples = 1
    np.random.seed(0)
    indices = np.random.permutation(num_total_samples)
    vtest_idx = indices[:num_test_samples]
    #print '{}'.format(vtest_idx)
    return vtest_idx

# =======================================================
# function to setup train/val dirs
# =======================================================
def setupTrainValDirs():
    # delete all existing (old) directories including sub-dirs first
    if os.path.exists(trainDir):
        shutil.rmtree(trainDir)
        print('removed %s' + trainDir)
    if os.path.exists(valDir):
        shutil.rmtree(valDir)
        print('removed %s' + valDir)
    # setup train sub-dirs
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    if not os.path.exists(taDir):
        os.makedirs(taDir)
    if not os.path.exists(tdDir):
        os.makedirs(tdDir)
    # setup validation sub-dirs
    if not os.path.exists(valDir):
        os.makedirs(valDir)
    if not os.path.exists(vaDir):
        os.makedirs(vaDir)
    if not os.path.exists(vdDir):
        os.makedirs(vdDir)

# ====================================================================
# Logic below aligns to these steps:
#       0. setup destination train/val dirs
#       1. for each file in merged/Annotations dir,
#       2. check if image_id is part of valIdx
#       3. move annotation and image files accordingly (to val/ or train/)
#       4. update list_files.txt
# ====================================================================

# setup output dirs
setupTrainValDirs()

# Opening the destination val & train foo_bar_list.txt files..."
trainListFileName = '%s/train_anno_image_list.txt'%(trainDir)
dstFile_trainList = open('%s/train_anno_image_list.txt'%(trainDir), 'w')
dstFile_trainImageList = open('%s/train_image_list.txt'%(trainDir), 'w')
dstFile_trainObjSizesList = open('%s/train_sizes_list.txt'%(trainDir), 'w')

valListFileName = '%s/val_anno_image_list.txt'%(valDir)
dstFile_valList = open('%s/val_anno_image_list.txt'%(valDir), 'w')
dstFile_valImageList = open('%s/val_image_list.txt'%(valDir), 'w')
dstFile_valObjSizesList = open('%s/val_sizes_list.txt'%(valDir), 'w')

# setup src annotations and image dirs as merged/Annotations and merged/prefixDir dirs
srcAnnoDir = dstAnnoDir
srcImageDir = dstImageDir

# walk through the files and make them as part of train or val
maFiles = list_files(srcAnnoDir)
numOfFiles = len(maFiles)

print('Number of Files in merged/Annotations dir: %s & newIdx: %s'%(numOfFiles, newIdx))

## ==============================================================================
#  count number of s/m/l objects in each category for val & train datasets
#  initialize the array based on number of labels in the labelmap file
## ==============================================================================
numOfLabels = len(name_to_coco_catid)             ## HACK ALERT XXX
#numOfLabels = len(name_to_coco_catid) + 1         ## TODO: COCO dataset has catid 90 assigned
print ('Num of Labels: %d\n'%(numOfLabels))

tsobj = {catid : 0 for catid in name_to_coco_catid.values()}
tmobj = {catid : 0 for catid in name_to_coco_catid.values()}
tlobj = {catid : 0 for catid in name_to_coco_catid.values()}
vsobj = {catid : 0 for catid in name_to_coco_catid.values()}
vmobj = {catid : 0 for catid in name_to_coco_catid.values()}
vlobj = {catid : 0 for catid in name_to_coco_catid.values()}

def countObjectSize(srcFile, isVal):
    #print('countObjectSize: srcFile : %s'%(srcFile))
    sarea = 32*32
    marea = 96*96
    # setup the right counters
    if (isVal == True):
        so = vsobj
        mo = vmobj
        lo = vlobj
    else:
        so = tsobj
        mo = tmobj
        lo = tlobj
    # account for each annotation in the passed file into val or train
    # setup input filr and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    anns = idata["annotation"]
    #change annotation data
    for i in range(len(anns)):
        area = int(anns[i]["area"])
        catid = int(anns[i]["category_id"])
        #print ('area: %d & catid: %d'%(area, catid))
        if (area < sarea):
            so[catid] += 1
        elif (area < marea):
            mo[catid] += 1
        else:
            lo[catid] += 1

## ==============================================================================

def hideAndSeekAugmentation(afile, inImageFile, hideNseek_class_list): 

    ifile = open(afile).read()
    idata = json.loads(ifile)
    anns = idata["annotation"]
    numAnns = len(anns)
    image = cv2.imread(inImageFile)

    # book keeping
    #numAugs = {catid : 0 for catid in name_to_coco_catid.values()}
    numAnnsInEachCat = [0]*len(name_to_coco_catid)
    imageIsChanged = False
    #print 'numAnns is {} and numAnnsInEachCat is {}'.format(numAnns, numAnnsInEachCat)

    # for each requested category, apply hide_n_seek annotations
    for i in range(numAnns):

        catId = int(anns[i]['category_id'])
        #print 'numAnnsInEachCat: {} and catId {}'.format(numAnnsInEachCat, catId)

        if str(catId) in hideNseek_class_list :

            naiec = numAnnsInEachCat[catId]
            # randomly apply a patch in the corner of bbox, to every other annotation in each category
            #if naiec % 2 == 0 : 
            if naiec >= 0 : 
                x, y, w, h = anns[i]["bbox"]
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                # default patch is x%  
                pw = int(w * 0.5)
                ph = int(h * 0.5)
                patch = np.zeros((ph, pw, 3), np.uint8)

                #to gray the RoI
                #gray_color = [255,255,255]
                #patch[np.where((patch == [0,0,0]).all(axis = 2))] = gray_color

                # apply this on one of the corners
                corner = (naiec + 3) % 4
                #print 'patch shape is {}, image shape is {}'.format(patch.shape, image.shape)
                #print 'patch pw:ph is {}:{}, anno bbox is {}'.format(pw, ph, anns[i]['bbox'])
                if corner == 0 : 
                    #apply at (x,y) corner of the bbox
                    image[y:y+ph, x:x+pw] = patch
                    #print 'corner {}, x is {}:{} and y is {}:{}'.format(corner, x,x+pw, y,y+ph)
                elif corner == 1 :
                    # apply at (x+w-pw, y) corner of the bbox
                    image[y:y+ph, x+w-pw:x+w] = patch
                    #print 'corner {}, x is {}:{} and y is {}:{}'.format(corner, x+w_pw,x+w, y,y+ph)
                elif corner == 2 :
                    # apply at (x, y+h-ph) corner of the bbox
                    image[y+h-ph:y+h,x:x+pw] = patch
                    #print 'corner {}, x is {}:{} and y is {}:{}'.format(corner, x,x+pw, y+h-ph,y+h)
                else :
                    # apply at (x+w-pw, y+h-ph) corner of the bbox
                    image[y+h-ph:y+h,x+w-pw:x+w] = patch
                    #print 'corner {}, x is {}:{} and y is {}:{}'.format(corner, x+w-pw,x+w, y+h-ph,y+h)

                # set flag to note image is changed
                imageIsChanged = True

            # increment numAnns in this category
            numAnnsInEachCat[catId] += 1

    if imageIsChanged : 
        # randomly apply a patch on the image outside of the annotations
        y = 5 + np.random.randint(50)
        x = 5 + np.random.randint(300)
        image[y:y+ph, x:x+pw] = patch
        y = 5 + np.random.randint(50)
        x = 300 + np.random.randint(50)
        image[y:y+ph, x:x+pw] = patch
        cv2.imwrite(inImageFile, image)


def dataAugmentationGaussianBlurRoI(afile, inImageFile):
    # default values for kernel size and sigma
    kernelSize = 5
    sigma = 2
    ifile = open(afile).read()
    idata = json.loads(ifile)
    anns = idata["annotation"]
    image = cv2.imread(inImageFile)
    # For every bbox(ROI) from the annotation file, add gb over the RoI
    numAnns = len(anns)
    #bboxes = np.zeros((numAnns, 4), dtype=int)
    contours = [[np.zeros((4, 2), dtype=int)]] * numAnns

    for i in range(numAnns):
        # setup the bbox & call blur rectangle method
        x, y, w, h = anns[i]["bbox"]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        contours[i] = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
    bimage = br.BlurContours(image, contours, kernelSize, sigma)
    cv2.imwrite(inImageFile, bimage)

def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):
# resize an image while preserving aspect ratio
# Give either hieght or width, not both.
# The image will resize to that dimension, and extrapolate the other
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def blackout_to_size(img, w, h):
# Returns a new image of size w by h.
# img is placed in the top left, the rest of the space will be black.
    black_image = np.zeros((h, w, 3), np.uint8)
    og_h, og_w = img.shape[0:2]
    black_image[0:og_h, 0:og_w] = img
    return black_image

def resizeImage(afile, inImageFile):
    imageToPredict = cv2.imread(inImageFile)
    with open(afile, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
        og_h, og_w = imageToPredict.shape[:2]
        # Determine which dimension can expand the least, and scale to that dimension
        img = None
        if float(resize_width) / og_w > float(resize_height) / og_h:
            img = image_resize(imageToPredict, height=resize_height)
        else:
            img = image_resize(imageToPredict, width=resize_width)
        # Figure out how much we just scaled by
        resized_h, resized_w = img.shape[:2]
        x_scale = float(resized_w) / og_w
        y_scale = float(resized_h) / og_h
        #print (x_scale, y_scale)
        img = np.array(img);

        data['image']['height'] = resize_height
        data['image']['width'] = resize_width

        # Scale each bbox by the scale factor
        for annotation in data["annotation"]:
            bx, by, bw, bh = annotation["bbox"]
            # Scale the bbox, clamp to image bounds.
            nx = max(0, int(np.round(bx * x_scale)))
            ny = max(0, int(np.round(by * y_scale)))
            nxmax = min(int(np.round((bx + bw) * x_scale)), resize_width - 1)
            nymax = min(int(np.round((by + bh) * y_scale)), resize_height - 1)
            nw = nxmax - nx
            nh = nymax - ny
            annotation["bbox"] = [nx, ny, nw, nh]
            annotation["area"] = nw * nh

        img = blackout_to_size(img, resize_width, resize_height)
    # Save to dst
    with open(afile, 'w') as f:
        json.dump(data, f, indent=2)
    cv2.imwrite(inImageFile, img)

# get indices for val split. Use newIdx as the total num of files
valIdx = create_test_idx((newIdx - 1), valSplit)
addGaussianBlur = False
numOfTrainSamples = (newIdx - 1) - len(valIdx)
factor = int(round(gbPercentage * 0.01 * numOfTrainSamples))

if factor is 0:
    factor = 1

#print 'Factor: ' + str(factor)
if not gbPercentage is 0:
    everyNthFrame = int(numOfTrainSamples/factor)
else:
    everyNthFrame = 1000000     # some large value
print "#Train Samples: %d, gb-percent: %d EveryNth: %d #val_samples: %d"%(numOfTrainSamples, gbPercentage, everyNthFrame, len(valIdx))

## ------------------------------------------------------------------------------------
#  check & return number of occurrences of the passed category Id in the inputfile
#
def checkIfCategoryIdExists(srcFile, catId):
    count = 0
    # setup input file and context
    ifile = open(srcFile).read()
    idata = json.loads(ifile)
    anns = idata["annotation"]
    image = idata["image"]
    # ignore hard-negatives
    #if 'hard_negative' in image and image['hard_negative'] is not None :
    if 'hard_negative' in image:
       return 1
    #check category Id
    for i in range(len(anns)):
        cid = int(anns[i]["category_id"])
        if cid == catId:
            count += 1
    return count
## ----------------------------------------------------------------------------------
trainCount = 0
movedFilesCount = 0
remDir = '%s/rem'%(mergeDir)
raDir = '%s/Annotations/'%(remDir)
rdDir = '%s/%s/'%(remDir, prefixDir)
# setup dir to move the removed files
if catId2check != 0:
    if os.path.exists(remDir):
        shutil.rmtree(remDir)
        print('removed %s' + remDir)
    # setup rem sub-dirs
    if not os.path.exists(raDir):
        os.makedirs(raDir)
    if not os.path.exists(rdDir):
        os.makedirs(rdDir)
# Create map of all names without extension to full paths to image file in merged data directory
# This isn't strictly necessary b/c the merged data directory is flat by construction, however
# this is cleaner and prepares the way for the future when we can perserve subdirectory file structure.
merged_image_filenames = list_files(dstImageDir)
merged_image_basenames = map(os.path.basename, merged_image_filenames)
merged_image_names_we = map(lambda b : os.path.splitext(b)[0], merged_image_basenames)
name_we_merged_image_filename_map = dict(zip(merged_image_names_we, merged_image_filenames))

for srcAnnoFile in maFiles:
    #print('srcAnnoFile : %s'%(srcAnnoFile))
    afilename, aext = os.path.splitext(srcAnnoFile)
    aPath, afname = os.path.split(srcAnnoFile)
    xfilename, xext = os.path.splitext(afname)
    image_id = int(xfilename.split('-')[-1])
    #print('image_id: %s & xfilename: %s & afilename: %s\n'%(image_id, xfilename, afilename))
    srcImageFile = name_we_merged_image_filename_map[xfilename]
    #print('srcImageFile: %s'%(srcImageFile))
    # get just the filenames
    xpath, afile = os.path.split(srcAnnoFile)
    xpath, ifile = os.path.split(srcImageFile)
    # weed out files with no requested categoryId into a separate dir
    if catId2check != 0:
        checkTotal = checkIfCategoryIdExists(srcAnnoFile, catId2check)
        if checkTotal is 0:
            #print "Moving files with NO annos for catId: %d"%(catId2check)
            movedFilesCount += 1
            dstAnnoFile = raDir + afile
            shutil.copy2(srcAnnoFile, dstAnnoFile)
            dstImageFile = rdDir + ifile
            if args.force_copy == True:
              shutil.copy2(srcImageFile, dstImageFile)
            else:
              os.symlink(srcImageFile, dstImageFile)
            continue
    # check if this needs to be part of train or val dataset
    valcheck = np.any(valIdx == image_id)
    if (valcheck == True):
        #print('imageId %s is part of validation set'%(image_id))
        #dstAnnoDir = vaDir
        #dstImageDir = vdDir
        dstAnnoFile = vaDir + afile
        dstImageFile = vdDir + ifile
        dstListFile = dstFile_valList
        dstImageListFile = dstFile_valImageList
    else:
        dstAnnoFile = taDir + afile
        dstImageFile = tdDir + ifile
        dstListFile = dstFile_trainList
        dstImageListFile = dstFile_trainImageList
        trainCount += 1
        if (trainCount % everyNthFrame is 0):
            addGaussianBlur = True

    shutil.copy2(srcAnnoFile, dstAnnoFile)
    # keep tab of s,m,l object sizes in each category
    countObjectSize(dstAnnoFile, valcheck)

    # If we're resizing the image, make a copy so the original is not affected.
    if resize_images:
        shutil.copy2(srcImageFile, dstImageFile)
    elif len(hideNseek_class_list) > 0 :
        shutil.copy2(srcImageFile, dstImageFile)
    elif addGaussianBlur is True:
        shutil.copy2(srcImageFile, dstImageFile)
    else:
        if args.force_copy == True:
          shutil.copy2(srcImageFile, dstImageFile)
        else:
          os.symlink(srcImageFile, dstImageFile)

    # do data augmentations
    if len(hideNseek_class_list) > 0 :
        hideAndSeekAugmentation(dstAnnoFile, dstImageFile, hideNseek_class_list)

    if addGaussianBlur is True:
        addGaussianBlur = False
        dataAugmentationGaussianBlurRoI(dstAnnoFile, dstImageFile)
    if resize_images:
        #print "dstAnno: " + dstAnnoFile
        resizeImage(dstAnnoFile, dstImageFile)
    dstListFile.write(dstImageFile)
    dstListFile.write(" ")
    dstListFile.write(dstAnnoFile)
    dstListFile.write("\n")
    dstImageListFile.write(dstImageFile)
    dstImageListFile.write("\n")

print "Number of Files with NO annos for catId: %d are %d"%(catId2check, movedFilesCount)
print "These are Moved into : %s"%(remDir)

##
# dump the count of object sizes
#
#       HACK ALERT XXX: indices for lmdata is limited to i%90 to work-around the issue that
#               our lables file has indices 0-89, where as coco uses id 90 also. Need to fix our side.
##
maxCatId = max([int(x) for x in name_to_coco_catid.values()])
print "maxCatId: %d" % maxCatId
for name, catid in name_to_coco_catid.items():
    if (tsobj[catid] > 0 or tmobj[catid] > 0 or tlobj[catid] > 0):
        #print('Train DS: category_Id %s : %d - %d %d %d'%(name, i,tsobj[catid], tmobj[catid], tlobj[catid]))
        tstr='%s(%d) %d %d %d'%(name, catid, tsobj[catid], tmobj[catid], tlobj[catid])
        dstFile_trainObjSizesList.write(tstr)
        dstFile_trainObjSizesList.write("\n")
    if (vsobj[catid] > 0 or vmobj[catid] > 0 or vlobj[catid] > 0):
        #print('Val DS:   category_Id %s : %d - %d %d %d'%(name, catid, vsobj[catid], vmobj[catid], vlobj[catid]))
        vstr='%s(%d) %d %d %d'%(name, catid, vsobj[catid], vmobj[catid], vlobj[catid])
        dstFile_valObjSizesList.write(vstr)
        dstFile_valObjSizesList.write("\n")
# close all open files
dstFile_valList.close()
dstFile_valImageList.close()
dstFile_valObjSizesList.close()

dstFile_trainList.close()
dstFile_trainImageList.close()
dstFile_trainObjSizesList.close()

# ==============================================================
# Create labelmap.prototxt, copy to destination.
# ==============================================================
mapfile = "%s/labelmap.prototxt" % mergeDir
labelmap_utils.create_labelmap_prototext(cocoCatIdsFileName, labelFile, mapfile)
dstLabelsJson = "%s/labelvector.json" % mergeDir
dstCatIdsJson = "%s/catids.json" % mergeDir
shutil.copy(cocoCatIdsFileName, dstCatIdsJson)
shutil.copy(labelFile, dstLabelsJson)
if args.remap_labels:
    dstRemapJson = "%s/anno_catids_remap.json" % mergeDir
    shutil.copy(args.remap_labels, dstRemapJson)

# ==========================================
# create LMDBs
#       Note: None of these are input args
# ==========================================
if skip_create_lmdb == True:
  print "Skipping creating LMDBs"
  print(" DONE! \n")
  exit(0)


extra_cmd = '--encode-type=jpeg --encoded --redo --shuffle'
lmdbScript = caffeRoot + '/scripts/create_annoset.py'
valLmdbDir = valDir + '/lmdb '
trainLmdbDir = trainDir + '/lmdb '

valArgs = valListFileName + ' ' + valLmdbDir + valDir + '/examples '
trainArgs = trainListFileName + ' ' + trainLmdbDir + trainDir + '/examples '

valInputArgs = '--anno-type=detection --label-type=json --label-map-file=' + mapfile + ' --min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label ' + extra_cmd + ' / ' + valArgs
trainInputArgs = '--anno-type=detection --label-type=json --label-map-file=' + mapfile + ' --min-dim=0 --max-dim=0 --resize-width=0 --resize-height=0 --check-label ' + extra_cmd + ' / ' + trainArgs

print('val cmds: %s %s \n'%(lmdbScript, valInputArgs))
print('train cmds: %s %s \n'%(lmdbScript, trainInputArgs))

cmd = 'python %s %s \n'%(lmdbScript, valInputArgs)
subprocess.call(cmd, shell=True)

cmd = 'python %s %s \n'%(lmdbScript, trainInputArgs)
subprocess.call(cmd, shell=True)

print(" Train LMDB dir is at: {}".format(trainLmdbDir))
print(" Val LMDB dir is at: {}".format(valLmdbDir))
print(" You can look at the LMDBs using $CAFFE_ROOT/scripts/data_prep/lmdbReader.py script, by providing the above directory full-path as input argument, one at a time")
print(" DONE! \n")
