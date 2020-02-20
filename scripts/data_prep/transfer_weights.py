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

parser = argparse.ArgumentParser()
parser.add_argument('--src_model', help='full path to source caffemodel file')
parser.add_argument('--src_proto', help='full path to source deploy.prototxt file')
parser.add_argument('--dst_model', help='full path to destination caffemodel file')
parser.add_argument('--dst_proto', help='full path to destination deploy.prototxt file')
parser.add_argument('--gpu', help='device number of gpu to use', type=int, default=0)
args = parser.parse_args()


caffe.set_device(args.gpu)
caffe.set_mode_gpu()

src_net = caffe.Net(args.src_proto, args.src_model, caffe.TEST)
dst_net = caffe.Net(args.dst_proto, args.dst_model, caffe.TEST)

for layer_name, param in src_net.params.iteritems():
    if len(param) == 2:
        print(layer_name + '\t' + str(param[0].data.shape) + str(param[1].data.shape))

        W = src_net.params[layer_name][0].data[...]
        b = src_net.params[layer_name][1].data[...]

        dst_net.params[layer_name][0].data[...] = W
        dst_net.params[layer_name][1].data[...] = b

    elif len(param) == 1:
        print(layer_name + '\t' + str(param[0].data.shape))
        W = src_net.params[layer_name][0].data[...]

        dst_net.params[layer_name][0].data[...] = W
    else:
        raise Exception('Unexpected parameter length')

dst_net.save(args.dst_model)



