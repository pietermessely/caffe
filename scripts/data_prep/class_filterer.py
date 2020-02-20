'''
    8/10/18 @ethomson
    Find all images that have certain classes,
    and move them to DST_DIR along with their annotations.
    Works only on coco style json annotations.
    Args:
        anno_src       : Folder containing all of your json annotations
        img_src        : Folder containing your source images
        dst_dir        : Location to put selected images in. It will make an 'Annotations' and 'Data' folder if they do not already exist
        wanted_classes : List of coco ids of classes you want to select. Ex: 1,6,67,94
        --ext          : Extension of your images. Defaults to .jpg
        --discard      : Remove all annotations that are not of one of your wanted classes.
        --mode         : 'or'  - If there's at least one annotation of ANY wanted class, accept the image.
                         'and' - If there's at least one annotation of ALL wanted classes, accept the image.
    Example run command:
        python class_filterer.py /mnt/data/flir-data/coco/COCO/Annotations/train2014/ /mnt/data/flir-data/coco/COCO/images/train2014/ /mnt/data/local_datasets/coco-ultimate/coco_no-hair-drier 1,2,3,4,5,6,7,8,9,10 --discard --mode or
'''

import json
import argparse
import shutil
import os
import copy
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("anno_src", type=str, help="Location of annotations")
parser.add_argument("img_src", type=str, help="Location of images")
parser.add_argument("dst_dir", type=str, help="Location to save selection")
parser.add_argument("wanted_classes", type=str, help="Comma separated list of coco ids wanted")
parser.add_argument("--ext", type=str, help="Image extension", default='.jpg')
parser.add_argument("--discard", action='store_true', help="Discard all annotations that are not in wanted_classes")
parser.add_argument("--mode", type=str, choices=["and", "or"], default='or', help="'and': One anno of ALL classes. \n 'or': One anno of ANY class.")
args = parser.parse_args()

SRC_DIR = args.anno_src
DST_DIR = args.dst_dir
IMG_SRC_DIR = args.img_src
IMG_EXT = args.ext
GUCCI_CLASSES = set([int(cid) for cid in args.wanted_classes.split(',')])
print "Accepted classes: "
print GUCCI_CLASSES
# or  : If there's at least one annotation of ANY gucci class, accept the image.
# and : If there's at least one annotation of ALL gucci classes, accept the image.
MODE = args.mode
discard_non_gucci = args.discard

if not os.path.exists(os.path.join(DST_DIR, 'Annotations')):
    os.makedirs(os.path.join(DST_DIR, 'Annotations'))
if not os.path.exists(os.path.join(DST_DIR, 'Data')):
    os.makedirs(os.path.join(DST_DIR, 'Data'))

count = 0
new_anno = {}

anno_files = os.listdir(SRC_DIR)
if MODE == "or":
    for file in anno_files:
        gucci_file = False
        print file
        with open(os.path.join(SRC_DIR, file), 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            if discard_non_gucci:
                new_anno = copy.deepcopy(data)
                new_anno['annotation'] = []
            for annotation in data["annotation"]:
                if int(annotation['category_id']) in GUCCI_CLASSES:
                    gucci_file = True
                    if discard_non_gucci:
                        new_anno['annotation'].append(annotation)
                    else:
                        break
            if gucci_file:
                count += 1
                print count
                if discard_non_gucci:
                    with open(os.path.join(DST_DIR, 'Annotations', file), 'w') as of:
                        json.dump(new_anno, of, indent=2)
                    shutil.copy(os.path.join(IMG_SRC_DIR, file.replace('.json', IMG_EXT)), os.path.join(DST_DIR, 'Data'))
                else:
                    shutil.copy(os.path.join(SRC_DIR, file), os.path.join(DST_DIR, 'Annotations'))
                    shutil.copy(os.path.join(IMG_SRC_DIR, file.replace('.json', IMG_EXT)), os.path.join(DST_DIR, 'Data'))
elif MODE == "and":

    has_classes = set()
    for file in anno_files:
        print file
        with open(os.path.join(SRC_DIR, file), 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            if discard_non_gucci:
                new_anno = copy.deepcopy(data)
                new_anno['annotation'] = []
            for annotation in data["annotation"]:
                if int(annotation['category_id']) in GUCCI_CLASSES:
                    has_classes.add(int(annotation['category_id']))
                    if discard_non_gucci:
                        new_anno['annotation'].append(annotation)
            if has_classes == GUCCI_CLASSES:
                count += 1
                print count
                if discard_non_gucci:
                    with open(os.path.join(DST_DIR, 'Annotations', file), 'w') as of:
                        json.dump(new_anno, of, indent=2)
                    shutil.copy(os.path.join(IMG_SRC_DIR, file.replace('.json', IMG_EXT)), os.path.join(DST_DIR, 'Data'))
                else:
                    shutil.copy(os.path.join(SRC_DIR, file), os.path.join(DST_DIR, 'Annotations'))
                    shutil.copy(os.path.join(IMG_SRC_DIR, file.replace('.json', IMG_EXT)), os.path.join(DST_DIR, 'Data'))
        has_classes.clear()
else:
    print "bad mode dood"

print "Number of images selected: {}".format(count)
