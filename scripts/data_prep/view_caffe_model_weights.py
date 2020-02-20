'''
Q3'2018 : kedar

Helps to visualize: the layer params / weights; layer blobs/data output after doing a 
forwrd pass with an image of choice. 

Caveat: This only works with caffe models. There are a couple of hard-coded paths in the code, that
            may need to be cleaned.

Adapted & Modified from : 
    https://github.com/smistad/visualize-caffe 
    http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

Inputs: 
    deploy_file     : e.g., deploy.prototxt
    model_file      : my-model-iter-10000.caffemodel
    layer_name      : layer you want to visualize e.g., conv1_1 in an ssd_vgg based network 
    --image_file    : Image you want to forward through the model prior to viewing the blobs, if none, uses a default image
    --color         : cmap color to use for visualization (e.g., gray, RdYlBu, Greens, Plasms, YlGnBu, YlGnBu_r, etc.,)

NOTE: The code that dumps the layers list along with output dimensions AND layer params(weights/bias) 
        with output dimensions is currently commented out. Only useful to get a dump for each type of model.

    e.g., 
    python view_caffe_model_weights.py /mnt/data/models/object_detectors/street-thermal/ssd_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/street-thermal/ssd_vgg_512x512/street-thermal_ssd_vgg_512x512_iter_20000.caffemodel conv1_1
            or
    python view_caffe_model_weights.py /mnt/data/models/object_detectors/street-thermal/ssd_vgg_512x512/deploy.prototxt /mnt/data/models/object_detectors/street-thermal/ssd_vgg_512x512/street-thermal_ssd_vgg_512x512_iter_20000.caffemodel conv1_1 --image_file=/home/kmadineni/work/crepo/caffe/examples/images/adas_ir_image.jpeg

'''

import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Make sure caffe can be found ... assuming this script is in data_prep
sys.path.append('../../python/')

import caffe

def normalize_array(inArray) : 
    # Normalize to 0-1
    min = inArray.min()
    max = inArray.max()
    inArray = (inArray - min) / (max - min)
    return inArray

def display_heat_map(net, layer_name) :

    padding = 4
    in_cmap = 'gray'

    cnvImage = np.copy(net.blobs[layer_name].data)
    weights = np.copy(net.params[layer_name][0].data)
    print 'cnvImage shape: {}, weights shape: {}'.format(cnvImage.shape, weights.shape)

    # N is the total number of convolutions
    N = weights.shape[0]*weights.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = weights.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    wresult = np.zeros((result_size, result_size))
    iresult = np.zeros((result_size, result_size))

    #print 'visualize_prams_or_blobs: N: %d, filters_per_row: %d, filter_szie: %d, result_size: %d'\
    #        %(N, filters_per_row, filter_size, result_size)

    # Tile image and weights into the results
    filter_x = 0
    filter_y = 0
    for n in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    wresult[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = weights[n, c, i, j]
                    iresult[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = cnvImage[n, c, i, j]
            filter_x += 1

    print 'wresult shape: {}, iresult shape: {}'.format(wresult.shape, iresult.shape)

    # Plot weights
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(wresult, cmap=in_cmap, interpolation='nearest')

    # Save plot if filename is set
    #if filename != '':
    #    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.title('weights')
    plt.show()

    # plot converted image
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(iresult, cmap=in_cmap, interpolation='nearest')
    plt.title('cnvImage')
    plt.show()



def visualize_params_or_blobs(net, layer_name, padding=4, filename='', isBlob=False, in_cmap='gray', imageFileName=''):

    if not imageFileName is '':
        cnvImage = np.copy(net.blobs[layer_name].data)
        weights = np.copy(net.params[layer_name][0].data)
        #display_heat_map(cnvImage, weights)

    if isBlob is True : 
        print 'visualizing blobs : converted image at layer_name: ' + str(layer_name)
        data = np.copy(net.blobs[layer_name].data)
    else :
        # The parameters are a list of [weights, biases]
        print 'visualizing params : weights at layer_name: ' + str(layer_name)
        data = np.copy(net.params[layer_name][0].data)

    print 'n=shape[0] is:  %d, c=shape[1] is: %d '%(data.shape[0], data.shape[1])

    # N is the total number of convolutions
    N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    print 'visualize_prams_or_blobs: N: %d, filters_per_row: %d, filter_szie: %d, result_size: %d'\
            %(N, filters_per_row, filter_size, result_size)

    # Tile data (image or weights) into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
            filter_x += 1

    print 'result shape: {}'.format(result.shape)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap=in_cmap, interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.title('As-Is')
    plt.show()

    result2 = result.copy()

    ## HACK to skip early
    return result2, N, filters_per_row

    # Normalize image to 0-1
    result = normalize_array(result)
    #min = result.min()
    #max = result.max()
    #result = (result - min) / (max - min)
    plt.imshow(result, cmap=in_cmap)
    plt.title('normalized to 0-1')
    plt.colorbar()
    plt.show()

    # threshold at 0.5
    result = result2.copy()
    result[result < 0.5] = 0
    plt.imshow(result, cmap=in_cmap)
    plt.title('threshold 0.5, Not Normalized 0-1')
    plt.colorbar()
    plt.show()

    # threshold to 0.3
    result = result2.copy()
    result[result < 0] = 0
    result /= np.max(result)
    result[result < 0.3] = 0
    plt.imshow(result, cmap=in_cmap)
    plt.title('threshold 0.3, Not Normalized')
    plt.colorbar()
    plt.show()

    # threshold to 0.7
    result = result2.copy()
    result[result < 0] = 0
    result /= np.max(result)
    result[result < 0.7] = 0
    plt.imshow(result, cmap=in_cmap)
    plt.title('threshold 0.7, Not Normalized')
    plt.colorbar()
    plt.show()


    return result2, N, filters_per_row

def show_net_layers(net):
    print '========= Layer Name \t Output Shape'
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

def show_net_params(net):
    print '========= Layer Names \t Params: weights, bias'
    for layer_name, param in net.params.iteritems():
        if len(param) is 2:
            print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
        else :
            print layer_name + '\t' + str(param[0].data.shape), 'Null'

def viewDetections (dout, imageFileName) :

    # Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
    numOfdetections = len(dout)
    elem = dout[0]
    print 'Num of detections %d size of each detection %d '%(numOfdetections, len(elem))
    frame = cv2.imread(imageFileName)
    oframe = frame
    rows = frame.shape[0]
    cols = frame.shape[1]
    for i in range(0,numOfdetections) :
        score = dout[i][2]
        if score > 0.39 :
            #print dout[i]
            catId = int(dout[i][1])
            x = int(dout[i][3] * cols)        
            y = int(dout[i][4] * rows)        
            xmax = int(dout[i][5] * cols)       
            ymax = int(dout[i][6] * rows)
            w = xmax - x       
            h = ymax - y
            displayStr = str(catId)
            cv2.rectangle(oframe, (x,y), (xmax, ymax), (255,0,0), 1)
            cv2.putText(oframe, str(displayStr), (x, y-2), 0, 0.3, (0,255,255))
        else :
            print 'Omitting detection: score: %.2f, catId: %d'%(score, int(dout[i][1]))

    cv2.imwrite('/tmp/dets.png', oframe)
    cv2.imshow('radek-view-weights',oframe)
    cv2.moveWindow('radek-view-weights', 100, 100) 

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow('radek-view-weights')
        pass

def visualize_heatmap(mapsTile, numOfMaps, mapsPerRow, filename='', imageFileName='', in_cmap='gray'):

    print 'visualizing heatmap'
    #heat_map = mapsTile[0][0]
    #heat_map = np.array(heat_map,dtype=np.float)
    #heat_map = cv2.resize(heat_map, (512,512))

    # Plot 
    if imageFileName is '': 
        print 'Need input image to show heatmap'
        return -1

    src = cv2.imread(imageFileName)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (512,512))

    plt.title('hm: src image')
    plt.imshow(src, cmap=in_cmap, interpolation='nearest')
    plt.colorbar()
    plt.show()
    #plt.imshow(heat_map, alpha = 0.5, cmap=in_cmap, interpolation='nearest')
    #plt.imshow(heat_map, alpha = 0.5, interpolation='nearest')
    y = 0
    x = 0
    if numOfMaps > 5:
        numOfMaps = 1

    for i in range (0, numOfMaps) : 
        x = i % mapsPerRow
        y = i / mapsPerRow
        heat_map = mapsTile[x][y]
        #print 'heat_map shape: {}'.format(heat_map.shape)
        #print heat_map

        #cv2.imshow('hm', heat_map)
        #cv2.moveWindow('hm', 100, 100) 
        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    cv2.destroyWindow('hm')
        #    pass

        heat_map = np.array(heat_map,dtype=np.float)
        heat_map = cv2.resize(heat_map, (512,512))
        #cv2.imshow('hm', heat_map)
        #cv2.moveWindow('hm', 100, 100) 
        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    cv2.destroyWindow('hm')
        #    pass

        plt.title('hm resized #: %d'%(i))
        plt.imshow(heat_map, cmap=in_cmap, interpolation='nearest')
        plt.colorbar()
        plt.show()

        plt.title('src + hm #: %d'%(i))
        #plt.imshow(heat_map, alpha=0.5, interpolation='nearest')
        #plt.imshow(heat_map, cmap=in_cmap, interpolation='nearest')
        #plt.imshow(src, cmap=in_cmap)
        #plt.imshow(heat_map, cmap=in_cmap)
        #print src.shape
        src2 = src[:, :, 0] + src[:, :, 1] + src[:, :, 2]
        #mergedImage = src[:, :, 0] + heat_map
        #print src.shape
        print src2.shape
        mergedImage = src2 + heat_map
        plt.imshow(mergedImage, cmap=in_cmap, interpolation='nearest')
        plt.colorbar()
        plt.show()

    # Save plot if filename is set
    #if filename != '':
    #    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_file", help="deploy file")
    parser.add_argument("model_file", help="model file")
    parser.add_argument("layer_name", help="layer name")
    parser.add_argument("--image_file", help="Input Image into model", default='')
    parser.add_argument("--cmap", help="cmap color to use for plotting", default='gray')
    parser.add_argument("--dump_layers", help="dump layer names and dimensions", default='')
    parser.add_argument("--view_weights", help="visualize weights", default='')
    parser.add_argument("--view_detections", help="visualize detections", default='')
    args = parser.parse_args()
	
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # Load model
    net = caffe.Net(args.deploy_file, args.model_file, caffe.TEST) 

    # setup some mean values for image
    mu = np.load('../../python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values: ', mu
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    net.blobs['data'].reshape(1, 3, 512,512) ## batch, 3-channel BGR images, image_size

    if not args.image_file is '' :
        imageFileName = args.image_file
        image = caffe.io.load_image(imageFileName)
    else:
        imageFileName = '../../examples/images/cat.jpg'
        image = caffe.io.load_image(imageFileName)

    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image) 
    #plt.axis('off')
    #plt.show()

    # copy image into net data layer & do a forward pass
    net.blobs['data'].data[...] = transformed_image
    detections = net.forward()

    # dump the CNN's layer name & output dimesions,  weights/bias dimenstions
    if not args.dump_layers is '' :
        show_net_layers(net)
        show_net_params(net)

    # visualize the requested layer params and data blobs 
    if not args.view_weights is '' :
        #mapsTile, numOfMaps, mapsPerRow = visualize_params_or_blobs(net, args.layer_name, filename='/tmp/vp.png', isBlob=False, in_cmap=args.cmap)
        visualize_params_or_blobs(net, args.layer_name, filename='/tmp/vp.png', isBlob=False, in_cmap=args.cmap, imageFileName=imageFileName)

    # view blobs 
    mapsTile, numOfMaps, mapsPerRow = visualize_params_or_blobs(net, args.layer_name, filename='/tmp/vb.png', isBlob=True, in_cmap=args.cmap, imageFileName=imageFileName)

    # view heatmap
    visualize_heatmap(mapsTile, numOfMaps, mapsPerRow, filename='/tmp/vh.png', imageFileName=imageFileName, in_cmap=args.cmap)
    #display_heat_map(net, args.layer_name)

    dout =  detections['detection_out'][0][0]
    if not args.view_detections is '' :
        viewDetections(dout, imageFileName)

    # Color map values are: 
    #   Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, 
    #   Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, 
    #   Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, 
    #   RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, 
    #   Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, 
    #   afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, 
    #   coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, 
    #   gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, 
    #   gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
    #   jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, 
    #   prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, 
    #   tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r

#def vis_square(data):
#    """Take an array of shape (n, height, width) or (n, height, width, 3)
#       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
#    
#    # normalize data for display
#    data = (data - data.min()) / (data.max() - data.min())
#    
#    # force the number of filters to be square
#    n = int(np.ceil(np.sqrt(data.shape[0])))
#    padding = (((0, n ** 2 - data.shape[0]),
#               (0, 1), (0, 1))                 # add some space between filters
#               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
#    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
#    
#    # tile the filters into an image
#    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#    
#    plt.imshow(data); plt.axis('off')
#    plt.show()

