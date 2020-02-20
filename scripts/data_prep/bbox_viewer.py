'''
7/24/18 @ethomson
    View annotation bboxes on images.
    Arguments:
        src_file  : text file with a list of images in a /Data folder. Should have corresponding /Annotations folder.
                    OR, if anno_type is txt, this should be a detection results file.
                    OR, if anno_type is conservator, this should be a big json file.
                    OR, if anno_type is lmdb, this should be the path to the lmdb directory. Ex: /path/to/merged/train/lmdb
        src_dir   : source directory with /Data and /Annotations folders containing images and annotations.
        ext       : file extension of the images
        save      : folder to save images with bboxes drawn on in. If not provided, does not save.
        anno_type : format of the annotations (json, xml, or txt).
                     * json        - coco style json annotations
                     * xml         - downloaded from conservator style
                     * txt         - ssd_detect output style. Use src_file to input txt file.
                     * conservator - one big json file in conservator metadata format. Use src_file to input json file.
                                  Use src_dir to specify image directory. Does not need to have /Data or /Annotations
                     * lmdb        - lmdb file viewer. Use src_file to input path to lmdb directory.
    Examples:
        python bbox_viewer.py --src_dir /mnt/fruitbasket/users/ethomson/datasets/sdpc/merged/
        python bbox_viewer.py --src_file /path/to/detection/results.txt --save /home/me/folder --anno_type txt -v -s 100
        python bbox_viewer.py --src_file /mnt/fruitbasket/untracked_datasets/syncity_oct_10/conservator_annos.json --anno_type conservator --src_dir /mnt/fruitbasket/untracked_datasets/syncity_oct_10/Data/ --ext .jpg
'''

import json
import cv2
import argparse
import fnmatch
import glob
import os
import re
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str, help=".txt file with list of images (fullpaths) or other input file.")
parser.add_argument("--src_dir", type=str, help="src dir with /Data and /Annotations")
parser.add_argument("--ext", type=str, help="image file extension", default=".jpeg")
parser.add_argument("--save", type=str, help="location to save images to. If not provided, does not save")
parser.add_argument("--anno_type", type=str, help="format of annotations", choices=["json", "xml", "txt", "conservator", "lmdb"], default="json")
parser.add_argument("-v", "--verbose", action='store_true', help="Print out image filepaths as they are displayed")
parser.add_argument("-s", "--skip", type=int, default=1, help="number of images to jump over when displaying.")
args = parser.parse_args()

src_file = args.src_file
src_dir = args.src_dir
ext = args.ext
save_loc = args.save
anno_type = args.anno_type
verbose = args.verbose
frame_skip = args.skip

# Error checking
if not src_dir and not src_file:
    raise Exception("Need either src_dir or src_file")
if src_dir and src_file and anno_type != "conservator":
    print "!! Both src_dir and image_list provided, but only src_dir will be used !!"
if anno_type == "txt" and (not src_file or src_dir):
    print "txt format requires an src_file, not a src_dir"

CID_MAP = {
    1  : ((255, 0, 0), "person"),
    2  : ((0, 255, 0), "bicycle"),
    3  : ((0, 0, 255), "car"),
    4  : ((0, 155, 255), "motorcycle"),
    6  : ((216, 105, 250), "bus"),
    8  : ((223, 243, 41), "truck"),
    15 : ((0, 255 ,255), "bird"),
    79 : ((100, 170, 220), "drone")
}
OTHER = ((157, 10, 120), "other")

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    s = os.path.splitext(os.path.basename(s))[0]
    #print(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

# Initialization
image_lst_file = []
if anno_type != "lmdb":
    if not src_dir:
        image_lst_file = open(src_file, 'r')
    else:
        image_lst_file = []
        img_search_dir = ""
        if anno_type != "conservator":
            img_search_dir = os.path.join(src_dir, 'Data')
        else:
            img_search_dir = src_dir
        for root, dirnames, filenames in os.walk(img_search_dir):
            for filename in fnmatch.filter(filenames, '*.*'):
                image_lst_file.append(os.path.join(root, filename))

        image_lst_file = [img_fname for img_fname in image_lst_file if os.path.splitext(img_fname)[1] in ['.jpg', '.png', '.tiff', '.jpeg', '.gif', '.bmp']]
        #img_name_lst = sorted([os.path.splitext(os.path.basename(img_path))[0] for img_path in image_lst_file], key=natural_sort_key)
        image_lst_file = sorted(image_lst_file, key=natural_sort_key)
        #print(image_lst_file[0:50])
else:
# Only import these if format is lmdb.
    import matplotlib
    import numpy as np
    import lmdb
    import caffe

# Display an image, and save if requested.
def show_and_save(location, filename, img):
    cv2.imshow("here a pic", img)
    if save_loc:
        cv2.imwrite(os.path.join(location, filename), img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        exit()

# Unique json files, coco style.
def display_json():
    for count, image_path in enumerate(image_lst_file):
        if count % frame_skip != 0:
            continue
        image_path = image_path.strip('\n')
        if verbose:
            print image_path
        image = cv2.imread(image_path)
        xmin, ymin, width, height, c_id = [None, None, None, None, None]
        with open(image_path.replace('Data', 'Annotations').replace(ext, '.json')) as f:
            data = json.load(f)
            for anno in data["annotation"]:
                bbox = anno["bbox"]
                c_id = anno["category_id"]
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                width = int(bbox[2])
                height = int(bbox[3])
                cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), CID_MAP.get(int(c_id), OTHER)[0] , 2)
                cv2.putText(image, CID_MAP.get(int(c_id), OTHER)[1], (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        show_and_save(save_loc, os.path.basename(image_path), image)
# Unique xml files, format of when dataset is dled from conservator.
def display_xml():
    for count, image_path in enumerate(image_lst_file):
        if count % frame_skip != 0:
            continue
        image_path = image_path.strip('\n')
        if verbose:
            print image_path
        image = cv2.imread(image_path)
        data_root = ET.parse(image_path.replace('Data', 'Annotations').replace(ext, '.xml')).getroot()
        for anno in data_root.iter('object'):
            #print anno.tag
            #print image_path.replace('Data', 'Annotations').replace(ext, '.xml')
            name = anno.find('name')
            if name is not None:
                c_id = name.text
                bbox = anno.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), CID_MAP.get(c_id, OTHER)[0] , 2)
                cv2.putText(image, CID_MAP.get(c_id, OTHER)[1], (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        show_and_save(save_loc, os.path.basename(image_path), image)
# One big txt file
# Each line should look like: image_id label score xmin ymin xmax ymax
def display_txt():
    with open(src_file, 'r') as detection_results:
        count = 0
        image = None
        prev_image_path = None
        for result in detection_results:
            image_path, c_id, _, xmin, ymin, xmax, ymax, _, _ = result.split(' ')
            # Draw all bboxes on an image before displaying it.
            if image_path != prev_image_path:
                count += 1
                if count % frame_skip != 0:
                    continue
                if verbose:
                    print image_path
                if image is not None:
                    show_and_save(save_loc, os.path.basename(image_path), image)
                image = cv2.imread(image_path)
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            c_id = int(c_id)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), CID_MAP.get(c_id, OTHER)[0] , 2)
            cv2.putText(image, CID_MAP.get(int(c_id), OTHER)[1], (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            prev_image_path = image_path
# One big json file. Conservator metadata format.
# https://www.flirconservator.com/help
def display_conservator():
    with open(src_file, 'r') as f:
        data = json.load(f)
        count = 0
        for image_path, frame in zip(image_lst_file, data["videos"][0]["frames"]) :
            count += 1
            if count % frame_skip != 0:
                continue
            if verbose:
                print image_path
            image = cv2.imread(image_path)
            xmin, ymin, width, height = [None, None, None, None]
            for anno in frame["annotations"]:
                bbox = anno["boundingBox"]
                xmin = int(bbox['x'])
                ymin = int(bbox['y'])
                width = int(bbox['w'])
                height = int(bbox['h'])
                c_id = 1
                cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), CID_MAP.get(int(c_id), OTHER)[0] , 2)
                cv2.putText(image, anno["labels"][0], (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            show_and_save(save_loc, os.path.basename(image_path), image)

# Lmdb file. src_file is the path to lmdb directory
def display_lmdb():
    if not os.path.exists(src_file):
        print "invalid lmdb file"
        exit()
    lmdb_env = lmdb.open(src_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.AnnotatedDatum()

    count = 0
    for key, value in lmdb_cursor:
        count += 1
        if count % frame_skip != 0:
            continue
        if verbose:
            print key
        datum.ParseFromString(value)
        data = datum.datum
        grp = datum.annotation_group

        arr = np.frombuffer(data.data, dtype='uint8')
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        height, width = img.shape[0:2]
        for annotation in grp:
            for bbox in annotation.annotation:
                # get the area of the bbox and print it (km)
                xmin = int(bbox.bbox.xmin * width)
                xmax = int(bbox.bbox.xmax * width)
                ymin = int(bbox.bbox.ymin * height)
                ymax = int(bbox.bbox.ymax * height)
                bwidth = xmax - xmin + 1
                bheight = ymax - ymin + 1
                barea = bwidth * bheight
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), CID_MAP.get(int(annotation.group_label), OTHER)[0])
                cv2.putText(img, CID_MAP.get(int(annotation.group_label), OTHER)[1], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

        show_and_save(save_loc, key, img)

formats = {
           "json" : display_json,
           "xml" : display_xml,
           "txt" : display_txt,
           "conservator" : display_conservator,
           "lmdb" : display_lmdb
           }
formats[anno_type]()
