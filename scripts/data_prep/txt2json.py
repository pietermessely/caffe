"""
Q2'2018:   create seprate resutls file based on detection sizes s,m & l
@author:   kedar m

Q1'2018:   Augmented to count size of each label/object as s/m/l from the detections
            output and write into the existing foo_bar_time_calc.json file.
@author:   kedar m

Q4 2017:   Created to convert the ssd_detect output txt to json.
@author:   kedar m

"""

'''
    This is to create a detections json file from the detection results txt file. 
    Note: conversion assumes txt file record/elements sequence 
'''
import os
import json
import shutil
import string
import sys
import argparse

# input arguments 
parser = argparse.ArgumentParser()
parser.add_argument("inputFile", type=str,
                    help="detection results text file. Absolute Path")

# get the input file name
args = parser.parse_args()
inputFile = args.inputFile

# check if the input file exists and or valid
print 'inputFile:{}'.format(inputFile)
if not os.path.exists(inputFile):
    print 'inputFile does NOT exist! {}'.format(inputFile)
    sys.exit("Invalid Input File")

# setup the output detection results filename for coco use and 
#       time_calc filename for our use : both are json files
origfile, oext = os.path.splitext(inputFile)
outfile = origfile + '.json'
#print 'outfile: {}'.format(outfile)

outfile2 = origfile + '_time_calc' + '.json'
#print 'outfile2: {}'.format(outfile2)


# output data record place holder
odata = []
odata2 = []
odata3 = []

odataSmall = []
odataMedium = []
odataLarge = []

## =========================================================================
#       setup to account the s/m/l objects we detect for each category
#       Note: array size pre-defined for now
## =========================================================================
sarea = 32*32
marea = 96*96
numOfLabels = 91
odata4 = []
sobj = [0 for i in range(numOfLabels)]
mobj = [0 for i in range(numOfLabels)]
lobj = [0 for i in range(numOfLabels)]

# read the input file, convert then into json format and dump into output file
with open(inputFile, "r") as infile:

    for line in infile.readlines() :

        row = line.strip("\n")
        drow = {}
        drow = row.split(' ')

        # setup the filename first
        filename = drow[0]
        #print filename

        # weed out the last record 
        if (filename == 'LastRecord') :

            #print 'Last Record : {}'.format(drow)
            totalTime = int(drow[1])
            numberOfImages = int(drow[2])
            numberOfDetections = int(drow[3])
            totalDetectionTime = int(drow[4])
            totalNetTime = int(drow[5])
            batchSize = int(drow[6])
            numBatches = int(drow[7])
            dps = int(drow[8])
            fps = int(drow[9])

            odata3.append( {'total_time': totalTime, 'number_of_images': numberOfImages, 'number_of_detections' : numberOfDetections, 'total_detection_time' : totalDetectionTime, 'total_net_time' : totalNetTime, 'batch_size' : batchSize, 'num_of_batches' : numBatches, 'detections_per_second' : dps, 'frames_per_second' : fps} )

            #print 'totalTime {} msec, numOfImages {}, numOfDets {}, totalDetTime {} totalNetTime {} msec'.format(totalTime, numberOfImages, numberOfDetections, totalDetectionTime, totalNetTime)

            continue

        # calculate the bbox parameters and the area 
        xmin = int(drow[3])
        ymin = int(drow[4])
        xmax = int(drow[5])
        ymax = int(drow[6])
 
        width   = ((xmax - xmin) + 1)
        height  = ((ymax - ymin) + 1)
        area    = (width * height)
        bbox    = [xmin, ymin, width, height]

        # These are in microSeconds
        netTime = int(float(drow[7]))
        detTime = int(float(drow[8]))

        #print 'xmin {} ymin {} width {} height {} '.format(xmin, ymin, width, height)
        #print 'area {} bbox {}'.format(area, bbox )

        ## filename format from conservator is below, with appended imageId.
        ## video-D6YuLc44qf4k62YWn-frame-000140-sQcRbCiwHfr6joJXY-12345.jpeg
        ## Use '-' as the separator and get the last element as imageId. This
        ## would not force the filename to be a specific format
        path, fullname = os.path.split(filename)
        fname, fext = os.path.splitext(fullname)
        #imageId = int(fname.split('-')[5])
        try:
            imageId = int(fname.split('-')[-1])
        except ValueError:
            print 'No Image ID in this filename : %s'%(filename)
            print 'Try to run the script with -h option set to 1'
            sys.exit("No Image IDs: Try to run the script with -h option set to 1")
        
        # get the categoryId and score also
        catId   = int(drow[1])
        score   = float(drow[2])

        # keep tab of the detected object sizes against each category
        if (area < sarea) :
            sobj[catId] += 1
            odataSmall.append( { 'image_id': imageId, 'category_id': catId, 'bbox': bbox, 'area': area, 'score': score , 'det_time': detTime, 'net_time': netTime })
        elif (area < marea) :
            mobj[catId] += 1
            odataMedium.append( { 'image_id': imageId, 'category_id': catId, 'bbox': bbox, 'area': area, 'score': score , 'det_time': detTime, 'net_time': netTime })
        else :
            odataLarge.append( { 'image_id': imageId, 'category_id': catId, 'bbox': bbox, 'area': area, 'score': score , 'det_time': detTime, 'net_time': netTime })
            lobj[catId] += 1

        #print 'imageId {} catId {}, score {}'.format(imageId, catId, score)
        odata.append( { 'image_id': imageId, 'category_id': catId, 'bbox': bbox, 'area': area, 'score': score , 'det_time': detTime, 'net_time': netTime })

        odata2.append( { 'image_id': imageId, 'category_id': catId, 'area': area, 'score': score , 'det_time': detTime, 'net_time': netTime } )

    ## update the counts into odata4
    i = 0
    overall = 0
    for i in range(numOfLabels) : 
        if (sobj[i] > 0 or mobj[i] > 0 or lobj[i] > 0) :
            odata4.append({'category_id': i, 'Small': sobj[i], 'Medium': mobj[i], 'Large': lobj[i]})
            total = sobj[i] + mobj[i] + lobj[i]
            overall += total
            print('category_id: {} & s,m,l detection counts: {},{},{} and Total {}'.format(i,sobj[i],mobj[i],lobj[i], total))
    print('txt2json: Total Number of Detections: {}'.format(overall))

with open(outfile, 'w') as ofile:
    json.dump(odata, ofile, indent=4, sort_keys=True, separators=(',',':'))
    #print json.dumps(odata, ofile, indent=4, sort_keys=True, separators=(',',':'))

with open(outfile2, 'w') as ofile2:
    keys = ['detection_time', 'total_time', 'detected_object_sizes']
    data = {key: None for key in keys}
    data ['detection_time'] = odata2
    data ['total_time'] = odata3
    data ['detected_object_sizes'] = odata4
    json.dump(data, ofile2, indent=4, sort_keys=True, separators=(',',':'))
    #print json.dumps(odata2, ofile2, indent=4, sort_keys=True, separators=(',',':'))


## create s/m/l json files and dump the corresponding detections
##      DISABLED by defualt
createSMLFiles = False
if createSMLFiles is True :

    ofpath, ofname = os.path.split(outfile)
    sofile = ofpath + '/small_' + ofname
    mofile = ofpath + '/medium_' + ofname
    lofile = ofpath + '/large_' + ofname

    with open(sofile, 'w') as ofile:
        json.dump(odataSmall, ofile, indent=4, sort_keys=True, separators=(',',':'))

    with open(mofile, 'w') as ofile:
        json.dump(odataMedium, ofile, indent=4, sort_keys=True, separators=(',',':'))

    with open(lofile, 'w') as ofile:
        json.dump(odataLarge, ofile, indent=4, sort_keys=True, separators=(',',':'))


