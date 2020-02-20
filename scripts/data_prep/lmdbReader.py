import sys
import os
import argparse
import shutil
import matplotlib
import numpy as np
import lmdb
#import cv2 

#caffe_root = '/mnt/data/ssd_clean/caffe/'
#caffe_root = '/home/flir/warea/crepo/caffe/'
caffe_root = os.environ['CAFFE_ROOT']
print "CAFFE_ROOT: %s" % caffe_root
caffe_python_path = "%s/python" % caffe_root
print "caffe_python_path: %s" % caffe_python_path
sys.path.insert(0, caffe_python_path)
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'
#sys.path.insert(0, '/usr/local/caffe/python')

print "before import caffe"
import caffe
print "after import caffe"
print "before import cv2"
import cv2
print "after import cv2"

# input arguments 
parser = argparse.ArgumentParser()
parser.add_argument("lmdbFile", type=str,
                    help="LMDB file to view")

# get the input file name
args = parser.parse_args()
lmdb_file = args.lmdbFile

# check if the input file exists and or valid
print 'input lmdbFile:{}'.format(lmdb_file)
if not os.path.exists(lmdb_file):
    print 'input lmdb_file does NOT exist! {}'.format(lmdb_file)
    exit

lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.AnnotatedDatum()
count = 0

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    data = datum.datum
    grp = datum.annotation_group

    arr = np.frombuffer(data.data, dtype='uint8')
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    width = img.shape[1]
    height = img.shape[0]
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
            strLabel = '%s, %s'%(str(annotation.group_label), str(barea))

            #cv2.rectangle(img, (int(bbox.bbox.xmin * width), int(bbox.bbox.ymin*height)), (int(bbox.bbox.xmax*width), int(bbox.bbox.ymax*height)), (0,255,0))
            #cv2.putText(img, str(annotation.group_label), (int(bbox.bbox.xmin * width), int(bbox.bbox.ymin * height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0))
            cv2.putText(img, strLabel, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    cv2.imshow('decoded image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
#    cv2.waitKey(0)
#    cv2.waitKey(1)
#    count = count + 1
#    name = '/mnt/data/videos/syncity/img'+ str(count) + '.png'
#    cv2.imwrite(name, img)
#writer.write(img)
#writer.release()
#print (count)
