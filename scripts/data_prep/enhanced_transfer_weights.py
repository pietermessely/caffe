from __future__ import print_function

# Make sure that caffe is on the python path:
import os

pythonPath = os.environ.get("PYTHONPATH")
if (pythonPath is None) :
    print ('Need to setup enviromental variable PYTHONPATH to proceed. Need caffe proto')
    raise Exception ('Need to setup enviromental variable PYTHONPATH to proceed')
else :
    print ('PYTHONPATH : {}'.format(pythonPath))

import caffe
import argparse

import numpy as np
import json
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--src_model', help='full path to source caffemodel file')
parser.add_argument('--src_proto', help='full path to source deploy.prototxt file')
parser.add_argument('--dst_model', help='full path to destination caffemodel file')
parser.add_argument('--dst_proto', help='full path to destination deploy.prototxt file')
parser.add_argument('--gpu', help='device number of gpu to use', type=int, default=0)

parser.add_argument('--square_anchor', help="Destination model uses square-aspect anchor boxes only", action="store_true")
parser.add_argument('--num_src_aspect', help="Number of aspect ratios of the source model's anchor boxes", type=int, default=3)

parser.add_argument('--src_labelvec', help='full path to source labelvector file')
parser.add_argument('--dst_labelvec', help='full path to destination labelvector file')
parser.add_argument('--allow_unmapped_labels', action='store_true', help='allow labels in your dst_labelvec that are not in your src_labelvec')

args = parser.parse_args()

caffe.set_device(args.gpu)
caffe.set_mode_gpu()

src_net = caffe.Net(args.src_proto, args.src_model, caffe.TEST)
#dst_net = caffe.Net(args.dst_proto, args.dst_model, caffe.TEST)
dst_net = caffe.Net(args.dst_proto, caffe.TEST)

square = args.square_anchor
src_aspect = args.num_src_aspect

src_labelvec = json.load(open(args.src_labelvec, 'r'))["values"]
dst_labelvec = json.load(open(args.dst_labelvec, 'r'))["values"]

allow_unmapped_labels = args.allow_unmapped_labels

src_cls = len(src_labelvec)
dst_cls = len(dst_labelvec)

def mapLabels(src_labelvec, dst_labelvec):
    src_dict = dict(zip(src_labelvec, range(len(src_labelvec))))

    cherry_picked = []
    for label in dst_labelvec:
        cherry_picked.append(src_dict.get(label, None))
        if cherry_picked[-1] is None:
            if not allow_unmapped_labels:
                raise Exception(label + " is not found in source label vector. To ignore this, rerun with '--allow_unmapped_labels'")
            else:
                print("WARNING: label: '{}' is not found in source label vector.".format(label))
    cherry_picked[0] = 0

    return cherry_picked

print("source # of classes: {}".format(src_cls))
print("destination # of classes: {}".format(dst_cls))
print("using square anchor boxes only: {}".format(square))

cherry_picked = mapLabels(src_labelvec, dst_labelvec)
print(cherry_picked)

for layer_name, param in src_net.params.iteritems():
    if len(param) == 2:
        print(layer_name + '\t' + str(param[0].data.shape) + str(param[1].data.shape))

        W = src_net.params[layer_name][0].data[...]
        b = src_net.params[layer_name][1].data[...]

        if "_mbox_conf" in layer_name:
            if square:
                if layer_name.rsplit('_', 2)[0] in {"conv4_3_norm", "conv5_3_norm", "fc7", "conv6_2", "inception_3b/output", \
                 "inception_4e/output", "inception_5b/output", "conv6_1"}:
                    dst_net.params[layer_name][0].data[...] = W[:len(W)/src_aspect]
                    dst_net.params[layer_name][1].data[...] = b[:len(b)/src_aspect]
                    continue

                for dst_id, src_id in enumerate(cherry_picked):
                    if src_id is None:
                        # mu = 0, sigma = 0.1
                        dst_net.params[layer_name][0].data[dst_id] = 0.1 * np.random.randn(W.shape[1], W.shape[2], W.shape[3])
                        dst_net.params[layer_name][1].data[dst_id] = 0.0
                        continue

                    try:
                        dst_net.params[layer_name][0].data[dst_id] = W[src_id]
                    except:
                        raise Exception(layer_name, W.shape, dst_net.params[layer_name][0].data.shape)
                    dst_net.params[layer_name][1].data[dst_id] = b[src_id]

            else:
                if layer_name.rsplit('_', 2)[0] in {"conv4_3_norm", "conv5_3_norm", "fc7", "conv6_2", "inception_3b/output_norm", \
                 "inception_4e/output_norm", "inception_5b/output", "conv6_1"}:
                    dst_net.params[layer_name][0].data[...] = W
                    dst_net.params[layer_name][1].data[...] = b
                    continue

                for dst_id, src_id in enumerate(cherry_picked):
                    for a in range(src_aspect):
                        if src_id is None:
                            # mu = 0, sigma = 0.1
                            dst_net.params[layer_name][0].data[a*dst_cls+dst_id] = 0.1 * np.random.randn(W.shape[1], W.shape[2], W.shape[3])
                            dst_net.params[layer_name][1].data[a*dst_cls+dst_id] = 0.0
                        else:
                            dst_net.params[layer_name][0].data[a*dst_cls+dst_id] = W[a*src_cls+src_id]
                            dst_net.params[layer_name][1].data[a*dst_cls+dst_id] = b[a*src_cls+src_id]

        elif "_mbox_loc" in layer_name:
            if square:
                dst_net.params[layer_name][0].data[...] = W[:len(W)/src_aspect]
                dst_net.params[layer_name][1].data[...] = b[:len(b)/src_aspect]
            else:
                dst_net.params[layer_name][0].data[...] = W
                dst_net.params[layer_name][1].data[...] = b

        else:
            try:
                dst_net.params[layer_name][0].data[...] = W
            except:
                dst_shape = dst_net.params[layer_name][0].data.shape
                print(W.shape)
                dst_net.params[layer_name][0].data[...] = W[:, :, :dst_shape[2], :dst_shape[3]]
            dst_net.params[layer_name][1].data[...] = b

    elif len(param) == 1:
        print(layer_name + '\t' + str(param[0].data.shape))
        W = src_net.params[layer_name][0].data[...]

        dst_net.params[layer_name][0].data[...] = W
    else:
        raise Exception('Unexpected parameter length')

dst_net.save(args.dst_model)
