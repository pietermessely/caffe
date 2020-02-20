'''
 This code creates an ensemble caffemodel by averaging the weights of 8
 training snapshots.  The ensemble model is generally more accurate than 
 any of the individiual snapshots. 

 A text file containing the paths of the 8 snapshots, and a deploy.prototxt file
 is required as an input argument. It should look something like this: 

/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_1000.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_1500.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_2000.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_2500.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_3000.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_4000.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_5000.caffemodel
/home/user/work/caffe/models/VGGNet/coco/refinedet_nov14_adas_full_no_frz/coco_refinedet_nov14_adas_full_no_frz_iter_6000.caffemodel
/home/user/work/caffe/jobs/VGGNet/coco/refinedet_nov14_adas_full_no_frz/deploy.prototxt

'''

from __future__ import print_function
import numpy as np
import os 
import argparse


caffeRoot = os.environ.get("CAFFE_ROOT")
if (caffeRoot is None) :
    print ("Need to setup CAFFE_ROOT. Cannot Proceed Further")
    raise Exception ("CAFFE_ROOT : Not Set")
    sys.exit()

pythonPath = os.environ.get("PYTHONPATH")
if (pythonPath is None) :
    print ('Need to setup enviromental variable PYTHONPATH to proceed. Need caffe proto')
    raise Exception ('Need to setup enviromental variable PYTHONPATH to proceed')
else :
    print ('PYTHONPATH : {}'.format(pythonPath))

paths = []
deploy_path = ''

def get_src_model_paths(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    for line in content:
        if 'caffemodel' in line:
            paths.append(line)
        if 'deploy.prototxt' in line:
            deploy_path = line

    if len(paths) < 8:
        print('expected to find 8 caffemodel paths in ', fname)
    if deploy_path == '':
        print('expected to find a deploy.prototxt file in ', fname)

    return paths, deploy_path


parser = argparse.ArgumentParser()
parser.add_argument('--model_list', help='text file containing full paths for 8 caffemodel files')
parser.add_argument('--ensemble_path', help='path for the ensembled caffemodel', default='ensemble.caffemodel')
parser.add_argument('--gpu', help='device number of gpu to use', type=int, default=0)
args = parser.parse_args()

paths, deploy_path = get_src_model_paths(args.model_list)

import caffe
caffe.set_device(args.gpu)
caffe.set_mode_gpu()

coco_net1 = caffe.Net(deploy_path, paths[0], caffe.TEST)
coco_net2 = caffe.Net(deploy_path, paths[1], caffe.TEST)
coco_net3 = caffe.Net(deploy_path, paths[2], caffe.TEST)
coco_net4 = caffe.Net(deploy_path, paths[3], caffe.TEST)
coco_net5 = caffe.Net(deploy_path, paths[4], caffe.TEST)
coco_net6 = caffe.Net(deploy_path, paths[5], caffe.TEST)
coco_net7 = caffe.Net(deploy_path, paths[6], caffe.TEST)
coco_net8 = caffe.Net(deploy_path, paths[7], caffe.TEST)

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load MS COCO model specs
file = open(deploy_path, 'r')

coco_netspec = caffe_pb2.NetParameter()
text_format.Merge(str(file.read()), coco_netspec)


for layer_name, param in coco_net1.params.iteritems():
    if len(param) == 2:
        print(layer_name + '\t' + str(param[0].data.shape) + str(param[1].data.shape))

        W1 = coco_net1.params[layer_name][0].data[...]
        b1 = coco_net1.params[layer_name][1].data[...]

        W2 = coco_net2.params[layer_name][0].data[...]
        b2 = coco_net2.params[layer_name][1].data[...]

        W3 = coco_net3.params[layer_name][0].data[...]
        b3 = coco_net3.params[layer_name][1].data[...]

        W4 = coco_net4.params[layer_name][0].data[...]
        b4 = coco_net4.params[layer_name][1].data[...]

        W5 = coco_net5.params[layer_name][0].data[...]
        b5 = coco_net5.params[layer_name][1].data[...]

        W6 = coco_net6.params[layer_name][0].data[...]
        b6 = coco_net6.params[layer_name][1].data[...]

        W7 = coco_net7.params[layer_name][0].data[...]
        b7 = coco_net7.params[layer_name][1].data[...]

        W8 = coco_net8.params[layer_name][0].data[...]
        b8 = coco_net8.params[layer_name][1].data[...]

        W = (W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8) / 8
        b = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8) / 8

        coco_net1.params[layer_name][0].data[...] = W
        coco_net1.params[layer_name][1].data[...] = b

    else:
        print(layer_name + '\t' + str(param[0].data.shape))
        W1 = coco_net1.params[layer_name][0].data[...]
        W2 = coco_net2.params[layer_name][0].data[...]
        W3 = coco_net3.params[layer_name][0].data[...]
        W4 = coco_net4.params[layer_name][0].data[...]
        W5 = coco_net5.params[layer_name][0].data[...]
        W6 = coco_net6.params[layer_name][0].data[...]
        W7 = coco_net7.params[layer_name][0].data[...]
        W8 = coco_net8.params[layer_name][0].data[...]

        W = (W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8) / 8

        coco_net1.params[layer_name][0].data[...] = W


coco_net1.save(args.ensemble_path)
