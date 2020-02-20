import csv
import sys
import os
import copy
import argparse
import cv2
import numpy as np
import json


## --------------------------------------------------------------------
#   Helper function
## --------------------------------------------------------------------

def find(lst, key, value):
    grp = []
    for i, dic in enumerate(lst):
        if dic[key] == value:
            grp.append(i)
    return grp

## --------------------------------------------------------------------
#   for each annotation in each image
#       show it to user
#       get input from user i.e., Occluded ? Y or N
#       update annotation with 'occlusion' meta data
#   Finally: write the updated annotations data into a new file in saveDir
## --------------------------------------------------------------------

def addMetaData(srcFile) : 

    with open(srcFile, 'r') as f_input:
        src_input = json.load(f_input)

    annotations = src_input['annotations']
    images = src_input['images']
    categories = src_input['categories']

    newAnns = copy.deepcopy(annotations)
    imageIds = []

    # get all the images
    for i in range(len(images)) : 
        imageIds.append(images[i]['id'])
    imageIds = list(set(imageIds))
    imageIds.sort()
    rangemax = len(imageIds)

    occluded = False  # visiblity os NOT 100% 
    truncated = False # Only part of the object is in frame
    outData = []      # output data for new file

    keys = ['images', 'annotations', 'categories']
    outData = {key: None for key in keys}
    outData['images'] = copy.deepcopy(images)
    outData['categories'] = copy.deepcopy(categories)


    # for each annotation in each image, get user input: occluded ? y or n
    for idx in range(rangemax) :

        # init 
        occluded = False
        truncated = False

        # get all annotations in this image
        imgId = imageIds[idx]
        index_gt = find(annotations, 'image_id', imgId)

        # get the image file name
        index_fname = find(images, 'id', imgId)
        tmp_var = images[int(index_fname[0])] 
        imageFileName = tmp_var['file_name']

        window = 'metaDataUpdate : {}'.format(imageFileName)

        for idx in index_gt : 

            # display each annotation, get input, update metadata
            frame = cv2.imread(imageFileName)
            in_anno = annotations[idx]
            out_anno = newAnns[idx]
            x = int(in_anno['bbox'][0])
            y = int(in_anno['bbox'][1])
            w = int(in_anno['bbox'][2])
            h = int(in_anno['bbox'][3])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
            display_str = "Object state: Press o: if occluded, t: if truncated, else, any key"
            cv2.putText(frame, str(display_str), (10, 10), 0, 0.5, (255,0,255))

            cv2.imshow(window, frame)
            cv2.moveWindow(window, 100, 100)
            inkey = cv2.waitKey(0) & 0xFF
            if inkey == ord('o') : 
                occluded = True
            elif inkey == ord('t') : 
                truncated = True
            else : 
                pass

            #if cv2.waitKey(0) & 0xFF == ord('q'):
            #    break
        
            cv2.destroyAllWindows()

            out_anno['occluded'] = occluded
            out_anno['truncated'] = truncated
            newAnns[idx] = out_anno
            #print 'oldAnns[{}] is {}'.format(idx, annotations[idx])
            #print 'newAnns[{}] is {}'.format(idx, newAnns[idx])

    path,fname = os.path.split(srcFile)
    file2save = path + '/meta_' + fname 
    print 'new GT file with metadata is: {}'.format(file2save)
    outData['annotations'] = newAnns
    #outData['annotations'] = copy.deepcopy(newAnns)
    with open(file2save, 'w') as outfile : 
        json.dump(outData, outfile, indent=4, sort_keys=True, separators=(',',':'))

    print 'update Meta data: DONE!'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, help="absolute path to Ground Truth json file")

    args = parser.parse_args()
    srcFile = args.src_file or None

    if not os.path.exists(srcFile) : 
        print 'src_file {} does NOT exists. Bailing out'.format(srcFile)
        sys.exit('src_file {} does NOT exists. Bailing out'.format(srcFile))
    #else :
    #    fname,fext = os.path.splitext(srcFile)
    #    #print 'src file extension is {}'.format(fext)
    
    addMetaData(srcFile)

    print 'DONE!'

################################################################################

'''
Q1-2019 kedar

Script to query and get input if a given annotation is occluded, truncated or none

This is to augment existing ground truth files with this meta data and then use
this when parsing the detection resutls to get more insights into the algorithm
performance. 

example: 
    python addMetaData2GtAnnotations.py --src_file /mnt/data/flir-data/testdata/junk/test/GT.json

    output will be in:  /mnt/data/flir-data/testdata/junk/test/meta_GT.json

'''

