'''
9/17 @ethomson
Convert annotations from coco format to conservator metadata format
args:
    src_dir : Path to folder containing the individual json files with coco style annotations.
    outfile : File to save the resulting metadata json in. Does not need to exist.
    --labelmap : Labelmap file, converts number ids to string labels.
        EX:
            [{"id": 1,
              "name": "person"}]
The metadata dictionary should be manually edited.
'''

import json
import os
import glob
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str, help="location of annotation files (individual .jsons)")
parser.add_argument("outfile", type=str, help="file to save resulting .json metadata")
parser.add_argument("--labelmap", type=str, help="map from coco id to name", default="/mnt/fruitbasket/users/ethomson/labels/flir_catids.json")
args = parser.parse_args()
src_dir = args.src_dir
outfile = args.outfile

metadata = OrderedDict([
            ('version', 1),
            ('overwrite', True),
            ('videos', [OrderedDict([
                            ('name', 'coco_cars'),
                            ('description', 'All images from coco train2014 that have at least one car.'),
                            ('owner', 'edtthomson@gmail.com'),
                            ('location', 'various'),
                            ('frames', [])
                        ])])
            ])
labelmap = json.load(open(args.labelmap, 'r'))
id_to_name = {}
for label in labelmap:
    id_to_name[label["id"]] = str(label["name"]).strip()

def make_bbox_dict(x1, y1, w, h):
# xmin, ymin, xmax, ymax
    bbox = OrderedDict([
            ('x', int(x1)),
            ('y', int(y1)),
            ('w', int(w)),
            ('h', int(h))
            ])
    return bbox

def valid_bbox(x, y, w, h):
    if int(x) < 0 or int(y) < 0:
        return False
    if int(w) <= 0 or int(h) <= 0:
        return False
    return True

def convert_bbox(anno):
    bbox = anno["bbox"]
    c_id = int(anno["category_id"])
    boundingBox = make_bbox_dict(*bbox)
    converted_anno = OrderedDict([('labels'      , [id_to_name[c_id]]),
                                  ('boundingBox' , boundingBox)
                     ])
    return converted_anno

def convert_image(img_root, frame_num):
    print img_root
    frame_annos = OrderedDict([
                    ('frameIndex'  , frame_num),
                    ('annotations' , [])
                  ])
    with open(img_root, 'r') as f:
        data = json.load(f)
        for anno in data["annotation"]:
            if not valid_bbox(*anno["bbox"]):
                continue
            frame_annos['annotations'].append(convert_bbox(anno))

    return frame_annos

anno_files = glob.glob(os.path.join(src_dir, '*.json'))

frames = []
for frame_num, file in enumerate(sorted(anno_files)):
    frames.append(convert_image(file, frame_num))
metadata['videos'][0]['frames'] = frames

with open(outfile, 'w') as of:
    json.dump(metadata, of, indent=2)
