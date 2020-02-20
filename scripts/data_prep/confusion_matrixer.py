'''
10/12/18 @ethomson
Adapted from: https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py

Script to generate a confusion matrix.

Arguments:
    log_file        : Full path to detection log file.
                      Per line format- image filename, class id, conf, xmin, ymin, xmax, ymax, time1(optional), time2(optional)
    anno_dir        : Full path to json annotations folder
    --print_classes : Which class columns to print. Default is print all classes.
    --labelvector   : Labelvector to use. Default flir_labelvector.json (has phantom)
    --iou_thr       : IoU Threshold for determining matches. Default 0.5
Example run command:
    $ python confusion_matrixer.py /mnt/fruitbasket/users/ethomson/models/refinedet/coco-ultimate/workspace/test_280HD_dpc_combined_v2_test_detectionResults_catId_0_dot4.txt /mnt/fruitbasket/users/ethomson/datasets/test_280HD_dpc_combined_v2/test/json/ --print_classes 1,3,79
'''

import os
import json
import glob
import argparse
import numpy as np

IOU_THRESHOLD = None
DET_LOG = ""
ANNO_DIR = ""
LABEL_VECTOR = ""

# Compute the IoU between two bboxes
def compute_iou(gt_bbox, dt_bbox):
    g_xmin, g_ymin, gw, gh = [int(coord) for coord in gt_bbox]
    d_xmin, d_ymin, dw, dh = [int(coord) for coord in dt_bbox]

    g_xmax = g_xmin + gw
    g_ymax = g_ymin + gh
    d_xmax = d_xmin + dw
    d_ymax = d_ymin + dh

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_image(num_categories, gt_bboxes, gt_classes, dt_bboxes, dt_classes):
    # +1 to each side of the matrix for gts without detections, and detections without gts
    confusion_matrix = np.zeros(shape=(num_categories + 1, num_categories + 1))
    matches = []

    # Find all ground truth - detection bbox matches with > 0.5 IoU
    for i in range(len(gt_bboxes)):
        for j in range(len(dt_bboxes)):
            iou = compute_iou(gt_bboxes[i], dt_bboxes[j])
            if iou >= IOU_THRESHOLD:
                matches.append([i, j, iou])
    matches = np.array(matches)

    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:,1], return_index=True)[1]]
        # Sort the list again by descending IOU.
        # Removing duplicates doesn't preserve our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:,0], return_index=True)[1]]

        for i in range(len(gt_bboxes)):
            if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                confusion_matrix[gt_classes[i] - 1][dt_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1
            # A ground truth without a matching detection, goes in last row.
            else:
                confusion_matrix[confusion_matrix.shape[1] - 1][gt_classes[i] - 1] += 1

        for i in range(len(dt_bboxes)):
            # A detection without a matching ground truth, goes in last column.
            if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                confusion_matrix[dt_classes[i] - 1][confusion_matrix.shape[0] - 1] += 1

    return confusion_matrix

# Given a list of detections for an image, return them in the format we want
# Given format:
#   [[class id, conf, xmin, ymin, xmax, ymax, time1(optional), time2(optional)], [...]]
# Returned format:
#   (detection bboxes, detection classes)
#   ([[x, y, w, h], [x, y, w, h], ...], [1, 79, ...])
def format_img_detections(det_list):
    dt_classes = [int(det[0]) for det in det_list]
    dt_bboxes = [det[2:6] for det in det_list]
    dt_bboxes = [[int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)] for x1, y1, x2, y2 in dt_bboxes]
    return (dt_bboxes, dt_classes)

# Given an annotation file, return formatted bboxes and class list
# Returned format:
#   (groundtruth bboxes, groundtruth classes)
#   ([[x, y, w, h], [x, y, w, h], ...], [1, 79, ...])
def format_img_gt(anno_fp):
    gt_bboxes = []
    gt_classes = []
    with open(anno_fp) as f:
        data = json.load(f)
        for annotation in data["annotation"]:
            gt_bboxes.append(annotation["bbox"])
            gt_classes.append(annotation["category_id"])
    return (gt_bboxes, gt_classes)

# Check for problems with the image dictionary
def check_image_dict(gt_bboxes, gt_classes, dt_bboxes, dt_classes):
    if gt_bboxes == None or gt_classes == None or dt_bboxes == None or dt_classes == None:
        return False
    return True

# Format and print the confusion matrix
# If c_ids is given, print matrix with only those categories
def display(confusion_matrix, class_list, c_ids=None):
    if not c_ids:
        c_ids = [i + 1 for i in range(len(class_list))]
    else:
        c_ids = [int(id) for id in c_ids.split(',')]
    # Add beggining and ending padding row
    c_ids.append(len(class_list)+ 1)
    c_ids.insert(0, 0)

    class_list.append("no match")
    # Add label row & column.
    disp_mat = np.zeros((len(class_list) + 1, len(class_list) + 1), object)
    disp_mat[1:, 1:] = confusion_matrix
    disp_mat[0, 1:] = class_list
    disp_mat[1:, 0] = class_list
    disp_mat[0, 0] = ''
    disp_mat = disp_mat[np.array(c_ids)[:, None], np.array(c_ids)]

    # Format and space nicely
    spacing = len(max(disp_mat[0], key=len))
    printer = np.vectorize(lambda x:'{0:{spacing}}'.format(x, spacing=spacing))
    print("\nConfusion Matrix:")
    print printer(disp_mat).astype(object)
    print "\n"
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to detection log file")
    parser.add_argument("anno_dir", type=str, help="Path to directory containing json annotations")
    parser.add_argument("--print_classes", type=str, help="Comma separated list of classes you wish to print. Default is all.",
                        default=None)
    parser.add_argument("--labelvector", type=str, help="Path to labelvector",
                        default="/mnt/fruitbasket/users/ethomson/labels/flir_labelvector.json")
    parser.add_argument("--iou_thr", type=float, help="IoU threshold for determing matches",
                        default=0.5)
    args = parser.parse_args()

    IOU_THRESHOLD = args.iou_thr
    ANNO_DIR = args.anno_dir
    DET_LOG = args.log_file
    LABEL_VECTOR = args.labelvector
    print_classes = args.print_classes

    # Don't shorten matrix printout, Fill lines entirely
    np.set_printoptions(threshold='nan', linewidth=1000)

    # Read labelvector
    with open(LABEL_VECTOR, 'r') as lv:
        data = json.load(lv)
        class_list = data["values"][1:]
        num_categories = len(class_list)

    # Initialize image dictionary
    # Format:
    #   {image_name: [gt_bboxes, gt_classes, dt_bboxes, dt_classes]}
    images = {}
    anno_fps = glob.glob(os.path.join(ANNO_DIR, '*.json'))
    for anno_fp in anno_fps:
        images[os.path.splitext(os.path.basename(anno_fp))[0]] = [None, None, None, None]

    detections = {}
    # Gather detections from detection log
    # Conglomerate detections into dictionary by image
    with open(DET_LOG, 'r') as detlog:
        for det in detlog:
            det_bits = det.split(' ')
            if det_bits[0] != 'LastRecord':
                img_name = os.path.splitext(os.path.basename(det_bits[0]))[0]
                if  detections.get(img_name, None):
                    detections[img_name].append(det_bits[1:])
                else:
                    detections[img_name] = [det_bits[1:]]

    # Add per image detections to image dictionary
    for det in detections:
        dt_bboxes, dt_classes = format_img_detections(detections[det])
        if images.get(det, None):
            images[det][2] = dt_bboxes
            images[det][3] = dt_classes

    # Add per image ground truths to image dictionary
    for anno_fp in anno_fps:
        gt_bboxes, gt_classes = format_img_gt(anno_fp)
        anno_name = os.path.splitext(os.path.basename(anno_fp))[0]
        if images.get(anno_name, None):
            images[anno_name][0] = gt_bboxes
            images[anno_name][1] = gt_classes

    cm = np.zeros(shape=(num_categories + 1, num_categories + 1))
    # Accumulate the confusion matrix for each image
    for count, img in enumerate(images):
        if not check_image_dict(*images[img]):
            #print("Skipped image %d, no matches" % (count))
            continue
        cm = cm + process_image(num_categories, *images[img])
        #print count

    # Display the confusion matrix
    display(cm, class_list, print_classes)
