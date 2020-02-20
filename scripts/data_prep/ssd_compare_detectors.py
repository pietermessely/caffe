import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

image_list = sorted(glob.glob('/home/jim/Desktop/ADAS_Training_Dataset_Thermal06/PreviewData/*/*.jpeg'))
#print len(image_list)

# Make sure that caffe is on the python path:
caffe_root = '/home/jim/work/caffe'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

save_count = 0

# load PASCAL VOC labels
labelmap_file = 'data/coco/labelmap_coco.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


model_def = '/home/jim/Desktop/j_raw/deploy.prototxt'
model_weights = '/home/jim/Desktop/street_thermal_best/refinedet_vgg_512x512.caffemodel'
degraded_weights = '/home/jim/Desktop/j_raw/coco_refinedet_oct14_j_raw_iter_800.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
degraded_net = caffe.Net(model_def,      # defines the structure of the model
                degraded_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)
degraded_net.blobs['data'].reshape(1,3,image_resize,image_resize)


def count_people(dets):

    person_count = 0

    det_label = dets[0,0,:,1]
    det_conf = dets[0,0,:,2]

    # Get detections with confidence higher than 0.4.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)

    for i in xrange(top_conf.shape[0]):
        label = int(top_label_indices[i])
        label_name = top_labels[i]

        if (label_name == 'person'):
            person_count += 1

        return person_count

def draw_detections(img, dets, dst_file):

    # Parse the outputs.
    det_label = dets[0, 0, :, 1]
    det_conf = dets[0, 0, :, 2]
    det_xmin = dets[0, 0, :, 3]
    det_ymin = dets[0, 0, :, 4]
    det_xmax = dets[0, 0, :, 5]
    det_ymax = dets[0, 0, :, 6]

    # Get detections with confidence higher than 0.4.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]

        display_txt = '' #''%s: %.2f' % (label_name, score)

        color = (255, 255, 255)
        if (label_name == 'person'):
            color = (0, 255, 0)
        elif label_name == 'car':
            color = (0, 0, 255)
        elif label_name == 'bicycle':
            color = (255, 0, 0)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    #cv2.imwrite(dst_file, img)

for img_path in image_list:

    image = caffe.io.load_image(img_path)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    degraded_net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']
    degraded_detections = degraded_net.forward()['detection_out']

    person_count = 0
    degraded_person_count = 0

    person_count = count_people(detections)
    degraded_person_count = count_people(degraded_detections)

    if person_count == None:
        person_count = 0
    if degraded_person_count == None:
        degraded_person_count = 0

    if (person_count != degraded_person_count):

        img8 = 255 * image  # Now scale by 255
        img = img8.astype(np.uint8)

        #clean_image = image
        dst_file = 'save/img_' + str(save_count) + '.png'
        draw_detections(img, detections, dst_file)

        img_degraded = img8.astype(np.uint8)
        degraded_dst_file = 'save/_img_' + str(save_count) + '_degraded.png'
        draw_detections(img_degraded, degraded_detections, degraded_dst_file)

        save_count += 1

        print(person_count, degraded_person_count)

        vis = np.concatenate((img, img_degraded), axis=1)
        cv2.imshow('images', vis)
        cv2.waitKey(1)

        cv2.imwrite(dst_file, vis)


