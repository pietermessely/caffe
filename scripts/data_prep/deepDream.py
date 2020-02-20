# imports and basic setup
import sys
import argparse
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import cv2

# Make sure caffe can be found ... assuming this script is in data_prep
sys.path.append('../../python/')

import caffe

mu_global = []

default_LastLayer = "conv6_2"       ## assuming base VGG-16 net's last conv layer
default_StepSize=1.5
default_Jitter=32
default_Iter_n=10
default_Octave_n=4
default_OctaveScale=1.4
default_clip=True

numOfDreamImagesToSave=4

inIter_n=10
inStepSize=1.2
inJitter=8        ## used for random shift of tiles - replaced with inOx, inOy
inOctave_n=3
inOctaveScale=1.4
inClip=False
inLastLayer="conv6_2"

## x,y co-ordinates from which the jitter/shift will be applied
inOx = 100      
inOy = 100

enableCv2Images = True
showImages = True
saveImages = False
filename2save = '/tmp/ddream.jpg'
modelNameTag = 'coco'

gimg_features = []


def showarray(a, fmt='jpeg'):
    aorig = a.copy()
    a = np.uint8(np.clip(a, 0, 255))
    #f = StringIO()
    #PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))
    #im = PIL.Image.open(f)
    #im.show()

    if enableCv2Images is True : 
        #a = (aorig - aorig.mean())/max(aorig.std(), 1e-4)*0.1 + 0.5
        #a = np.uint8(np.clip(a, 0, 255))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        #a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        if saveImages :
            cv2.imwrite(filename2save, a)
        if showImages:
            #cv2.resize(a, (1024,1024))
            cv2.imshow('image',a)
            cv2.moveWindow('image',100,200)
            if cv2.waitKey(0) & 0xFF == ord('q') :
                cv2.destroyWindow('image')
            cv2.destroyAllWindows()

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    if False : 
        #return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
        return np.float32(np.rollaxis(img, 2)[::-1]) - mu_global
    else :
        im = np.float32(img)
        im = im[:,:,::-1]
        im -= np.array([104,117,123], dtype=np.float32)
        #im -= np.array([117,117,117], dtype=np.float32)
        im *= 1.0
        im = im.transpose((2,0,1)) ##BGR
        return im

def deprocess(net, img):
    if False : 
        #return np.dstack((img + net.transformer.mean['data'])[::-1])
        return np.dstack((img + mu_global))
    else :
        im = img.transpose(1,2,0)
        im /= 1.0
        im += np.array([104,117,123], dtype=np.float32)
        #im += np.array([117,117,117], dtype=np.float32)
        im = im[:,:,::-1] # RGB
        return im


def objective_L2(dst):
    dst.diff[:] = dst.data 

def objective_guide(dst):
    guide_features = gimg_features
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

def make_step(net, step_size=1.5, end=default_LastLayer, 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ## following 2 lines are original
    ox, oy = inOx, inOy  #ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    ## following 2 copied from keras-tf code
    #ox, oy = np.random.randint(-jitter, jitter+1, 2)
    #src.data[0] = np.roll(np.roll(src.data[0], ox, 1), oy, 0) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    ## original
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    ## from keras-tf implementation
    #src.data[0] = np.roll(np.roll(src.data[0], -ox, 1), -oy, 0) # unshift image
            
    if clip:
        bias = 104
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end=default_LastLayer, clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))

        print octave, i, end, vis.shape
        showarray(vis)
        #clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_file", help="deploy prototxt file")
    parser.add_argument("model_file", help="caffemodel file")
    parser.add_argument("--last_layer", help="last layer from which the output will be used", default='')
    parser.add_argument("--image_file", help="Input Image into model as guide image", default='')
    parser.add_argument("--num_octaves", help="number of gradient ascent steps", type=int, default=3)
    parser.add_argument("--x_val", help="x coordinate at which jitter will be applied", type=int, default=100)
    parser.add_argument("--y_val", help="y coordinate at which jitter will be applied", type=int, default=100)
    parser.add_argument("--name_tag", help="tag to use in the dream image filename", default='coco')
    parser.add_argument("--save_images", help="1: save images, 0: default, don't", type=int, default=0)
    parser.add_argument("--num_iters", help="10-100 Number of jitter iterations", type=int, default=10)
    args = parser.parse_args()

    # select GPU device if multiple devices exist
    caffe.set_mode_gpu()
    caffe.set_device(0) 

    net_fn   = args.deploy_file
    param_fn = args.model_file

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    tmpPrototxtFile = '/tmp/tmp.prototxt'
    open(tmpPrototxtFile, 'w').write(str(model))

    #clfr_net = caffe.Classifier(tmpPrototxtFile, param_fn,
    #                   mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
    #                   channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    net = caffe.Net(tmpPrototxtFile, param_fn, caffe.TEST)
    mu = np.load('../../python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values: ', mu
    mu_global = mu

    net.blobs['data'].reshape(1, 3, 512,512) ## batch, 3-channel BGR images, image_size
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


    #print "========= Net Keys ========="
    #print net.blobs.keys()
    #print "========= Net Keys ========="

    if not args.image_file is '' :
        gimageFileName = args.image_file
    else :
        gimageFileName = '../../examples/images/cat.jpg'
    
    imageFileName = '/tmp/noise.jpg'
    img_noise = np.random.uniform(size=(512,512,3)) + 100.0
    #img_noise = np.random.uniform(size=(768,1024,3)) + 100.0
    PIL.Image.fromarray(np.uint8(img_noise)).save(imageFileName)

    image = caffe.io.load_image(imageFileName)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    img = np.float32(PIL.Image.open(imageFileName))

    #global inLastLayer, default_LastLayer, inIter_n, inStepSize, inJitter, inOctave_n, inOctaveScale
    # setup last layer
    if not args.last_layer is '':
        inLastLayer = args.last_layer
    else :
        print "Last Layer is not provided. Using default {}".format(default_LastLayer)
        inLastLayer = default_LastLayer

    ## setup input x,y co-ordinates also
    inOx = args.x_val
    inOy = args.y_val

    inOctave_n = args.num_octaves

    print "----------- Last Layer is {}, inOx {}, inOy {} num_octaves {} --------------" \
            .format(inLastLayer, inOx, inOy, inOctave_n)

    # generate a couple of dream images
    print '------------ using L2 objective, with noise image -------------'
    imgOut=deepdream(net, img, end=inLastLayer, clip=inClip, iter_n=inIter_n, step_size=inStepSize, jitter=inJitter, octave_n=inOctave_n, octave_scale=inOctaveScale)

    # setup guided image features first
    print '------------ using defined objective with guided image -------------'
    gimg = np.float32(PIL.Image.open(gimageFileName))
    h, w = gimg.shape[:2]
    src, dst = net.blobs['data'], net.blobs[inLastLayer]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, gimg)
    net.forward(end=inLastLayer)
    gimg_features = dst.data[0].copy()

    # call deepdream with guided objective function
    img = np.float32(PIL.Image.open(imageFileName))

    gimgOut=deepdream(net, img, end=inLastLayer, clip=inClip, iter_n=inIter_n, step_size=inStepSize, jitter=inJitter, octave_n=inOctave_n, octave_scale=inOctaveScale, objective=objective_guide)

    #print net.blobs.keys()
    #net.blobs.keys()

    # turn-off image display
    showImages=True
    saveImages=args.save_images     #saveImages=False

    # setup output filenames
    frame = gimg        ## using guide image
    frame_i = 0

    # get the model name tag to use in the file name
    modelNameTag = args.name_tag

    # get the number of iters to do 
    inIter_n=args.num_iters

    # feed the output of the network back into it to generate more images
    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(numOfDreamImagesToSave):
        filename2save="/tmp/dream_%s_%s_x%d_y%d_niter%d_octv%d_%02d.jpg"%(modelNameTag, inLastLayer,inOx,inOy,inIter_n,inOctave_n,frame_i)
        print ' ======== deep dream image : ========= ' + filename2save
        frame = deepdream(net, frame, end=inLastLayer, clip=inClip, iter_n=inIter_n, step_size=inStepSize, jitter=inJitter, octave_n=inOctave_n, octave_scale=inOctaveScale)

        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1

    if saveImages:
        print "dream files are in /tmp/dream_*.jpg"

    print 'waky, waky - deep dream is done'

'''
Q4-2018 kedar

# Copied from https://github.com/google/deepdream/blob/master/dream.ipynb 
# and modified for use with object detectors

example:
python deepDream.py /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/coco_refinedet_vgg16_512x512_final.caffemodel --image_file=/home/kmadineni/Downloads/sky_small.jpg --last_layer=
    
python deepDream.py /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/street-thermal/refinedet_vgg_512x512/refinedet_vgg_512x512.caffemodel --image_file= --last_layer= 

for guided image:
    python deepDream.py /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/street-thermal/refinedet_vgg_512x512/refinedet_vgg_512x512.caffemodel --image_file=/home/kmadineni/work/crepo/caffe/examples/images/adas_sstreet_ir_real_1.jpg --last_layer= 

python deepDream.py /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/coco/refinedet_vgg_512x512/coco_refinedet_vgg16_512x512_final.caffemodel --image_file=/home/kmadineni/work/images/noise.jpg --last_layer=conv5_2 --x_val=100 --y_val=100 --name_tag=coco_gimg_noise --save_images=0 --num_iters=5 --num_octaves=3

'''

