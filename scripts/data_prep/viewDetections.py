import csv
import sys
import os
import shutil
import argparse
import cv2
import numpy as np
import json


## --------------------------------------------------------------------
#   handle data file with FID scores
## --------------------------------------------------------------------

# process 'batch' files at a time. 
def display_fid_batch_data(srcFile, saveDir, numFidImages2save, showImages) : 

    # Note: numFidImages2save == numBatchesToSave ? 
    src_data = {}
    numBestFidImagesSaved = 0
    numBatchesSaved = 0

    with open(srcFile, 'r') as f_input:
        src_input = json.load(f_input)

        # if sort has same values, preserves rows using field 0, which is the image_file_name
        reverse = False
        #sort_column = 'batch'
        sort_column = 'fid'
        print 'sort_column is {} and reverse is {}'.format(sort_column, reverse)
        src_data = sorted(src_input, key=lambda row: (row[sort_column], row[sort_column]),reverse=reverse)
    
    print 'Going through {} records and saving {} best batches'.format(len(src_data), numFidImages2save)

    if not saveDir is None : 
        pairSaveDir = saveDir + '/pairs'
        if os.path.exists(pairSaveDir) : 
            print 'clearing dir {}'.format(pairSaveDir)
            shutil.rmtree(batchDir)
        os.makedirs(pairSaveDir)

    # display each row from the sorted data
    prevBatch = 9999
    currBatch = 0

    for row in src_data : 
        fidScore = row['fid']
        file1 = row['file1']
        file2 = row['file2']
        currBatch = row['batch']

        win_name = 'Batch:{} & FID Score:{}'.format(currBatch, fidScore)
        image1 = cv2.imread(file1)
        image2 = cv2.imread(file2)

        # resize images to 640x512 always
        w = 640
        h = 512
        image1 = cv2.resize(image1, (w,h))
        cv2.putText(image1, file1, (10,20), 0, 0.3, (255,255,255))
        image2 = cv2.resize(image2, (w,h))
        cv2.putText(image2, file2, (10,20), 0, 0.3, (255,255,255))

        # stack the images
        image_to_show = np.hstack((image1, image2))
        cv2.putText(image_to_show, win_name, (10,35), 0, 0.3, (255,255,255))

        # save the stacked image if requested into the dir and using the src_file_name and prefix
        if not saveDir is None : 
            file2save = pairSaveDir + '/batch_' + str(currBatch) + '_fid_' + str(int(fidScore)) + '_' + os.path.basename(file1) 
            #print 'file2save is: {}'.format(file2save)
            cv2.imwrite(file2save, image_to_show)

            # also save the src image in a separate dir based on batch number, upto to numFidImages2Save
            if currBatch != prevBatch : 
                prevBatch = currBatch
                batchDir = saveDir + '/batch' + str(currBatch) + '-fid-' + str(int(fidScore))
                if os.path.exists(batchDir) : 
                    print 'clearing batch dir to save images {}'.format(batchDir)
                    shutil.rmtree(batchDir)
                s1batchDir = batchDir + '/set1'
                s2batchDir = batchDir + '/set2'
                print 'creating dir(s) to save images based on fid scores and batches {}'.format(batchDir)
                print '{} : {}'.format(s1batchDir, s2batchDir)
                os.makedirs(batchDir)
                os.makedirs(s1batchDir)
                os.makedirs(s2batchDir)

            if numBestFidImagesSaved < numFidImages2Save :
                s1_image2save = s1batchDir + '/' + os.path.basename(file1)
                os.symlink(file1, s1_image2save)
                s2_image2save = s2batchDir + '/' + os.path.basename(file2)
                os.symlink(file2, s2_image2save)
                numBestFidImagesSaved += 1

        if showImages : 
            cv2.imshow(win_name, image_to_show)

            cv2.moveWindow(win_name, 100, 100)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
            cv2.destroyAllWindows()

    print 'sorting and bookkeeping of batched files based on fid is DONE!'


# one file at a time
def display_fid_data(srcFile, saveDir, numFidImages2save, showImages) : 

    src_data = {}
    numBestFidImagesSaved = 0

    if not saveDir is None : 
        bestFidImagesDir = saveDir + '/bestFidImages'
        if os.path.exists(bestFidImagesDir) : 
            print 'clearing dir to save images {}'.format(bestFidImagesDir)
            shutil.rmtree(bestFidImagesDir)
        print 'creating dir to save images with best fid scores {}'.format(bestFidImagesDir)
        os.makedirs(bestFidImagesDir)

        pairSaveDir = saveDir + '/pairs'
        if os.path.exists(pairSaveDir) : 
            print 'clearing dir {}'.format(pairSaveDir)
            shutil.rmtree(batchDir)
        os.makedirs(pairSaveDir)

    with open(srcFile, 'r') as f_input:
        src_input = json.load(f_input)

        # if sort has same values, preserves rows using field 0, which is the image_file_name
        reverse = False
        sort_column = 'fid'
        print 'sort_column is {} and reverse is {}'.format(sort_column, reverse)
        src_data = sorted(src_input, key=lambda row: (row[sort_column], row[sort_column]),reverse=reverse)
    
    print 'Going through {} records and saving {} best images'.format(len(src_data), numFidImages2save)

    # display each row from the sorted data
    for row in src_data : 
        fidScore = row['fid']
        file1 = row['file1']
        file2 = row['file2']

        win_name = 'FID Score: {}'.format(fidScore)
        image1 = cv2.imread(file1)
        image2 = cv2.imread(file2)

        # resize images to 640x512 always
        w = 640
        h = 512
        image1 = cv2.resize(image1, (w,h))
        cv2.putText(image1, file1, (10,20), 0, 0.3, (255,255,255))
        image2 = cv2.resize(image2, (w,h))
        cv2.putText(image2, file2, (10,20), 0, 0.3, (255,255,255))

        # stack the images
        image_to_show = np.hstack((image1, image2))
        cv2.putText(image_to_show, win_name, (10,35), 0, 0.3, (255,255,255))

        # save the stacked image if requested into the dir and using the src_file_name and prefix
        if not saveDir is None : 
            #file2save = saveDir + '/fid_pair_' + os.path.basename(file1) 
            file2save = pairSaveDir + '/fid_' + str(int(fidScore)) + '_' + os.path.basename(file1) 
            #print 'file2save is: {}'.format(file2save)
            cv2.imwrite(file2save, image_to_show)

            # also save the src image in a separate dir, upto to numFidImages2Save
            if numBestFidImagesSaved < numFidImages2Save :
                image2save = bestFidImagesDir + '/' + os.path.basename(file1)
                os.symlink(file1, image2save)
                numBestFidImagesSaved += 1

        if showImages : 
            cv2.imshow(win_name, image_to_show)

            cv2.moveWindow(win_name, 100, 100)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
            cv2.destroyAllWindows()

    print 'sorting and bookkeeping of fid pairs: DONE!'


## --------------------------------------------------------------------
#   handle csv data file
## --------------------------------------------------------------------

def display_csv_data(srcFile, compareFile, sort_column) : 

    compare = False
    if not compareFile is None : 
        cfile_input = open(compareFile, 'r')
        cfile_data = csv.DictReader(cfile_input)
        fields = cfile_data.fieldnames
        cmp_data = sorted(cfile_data, key=lambda row: (row[fields[sort_column]], row[fields[0]]))
        compare = True

    with open(srcFile, 'r') as f_input:
        src_input = csv.DictReader(f_input)
        fields = src_input.fieldnames
        # if sort has same values, preserves rows using field 0, which is the image_file_name
        src_data = sorted(src_input, key=lambda row: (row[fields[sort_column]], row[fields[0]]),reverse=True)
    
    # use openCV and start dispalying these files in the sorted order
    for row in src_data :
        image_file_name = row['image_file_name']
        image_id = row['image_id']

        win_name = 'viewDets : Image Id {}'.format(image_id)
        image_to_show = cv2.imread(image_file_name)

        if compare is True : 
            #cmp_row = [c_row for c_row in cmp_data if c_row['id'] == image_id][0]
            rec2cmp = [c_row for c_row in cmp_data if c_row['id'] == image_id]
            if rec2cmp : 
                cmp_row = rec2cmp[0]
                #print 'image_id is {} and cmp_row is {}'.format(image_id, cmp_row)
                cmp_image_file_name = cmp_row['image_file_name']
                cimage = cv2.imread(cmp_image_file_name)
                np_stack_h = np.hstack((image, cimage))
                np_concat_h = np.concatenate((image, cimage), axis=1)
                image_to_show = np_concat_h

        cv2.imshow(win_name,image_to_show)
        cv2.moveWindow(win_name, 100, 100)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
        cv2.destroyAllWindows()

    print 'csv: DONE!'

## --------------------------------------------------------------------
#   find overlapping bbox with the highest IOU and return it
## --------------------------------------------------------------------

def findOverlappedBbox(in_bbox, dts, type_of_det) : 

    ix, iy, iw, ih = in_bbox
    gt_area = float(iw*ih)

    bestMatchBbox = []
    bestMatchIoU = 0.0
    dtstr = '0'

    for item in dts : 
        if item['isFP'] is True and type_of_det == 'fp' :
            a,b,c,d = item['bbox']
            dx = int(a)
            dy = int(b)
            dw = int(c)
            dh = int(d)
            dt_area = float(dw*dh)

            # check for overlap with in_bbox. Stash the best one
            w = min(dx+dw, ix+iw) - max(dx, ix)
            h = min(dy+dh, iy+ih) - max(dy, iy)
            if w <= 0. or h <= 0. :
                continue

            intersection_area = float(w*h)
            total_area = float(gt_area + dt_area - intersection_area)

            if total_area <= 0 : 
                continue

            iou = intersection_area/total_area
            if iou > bestMatchIoU : 
                bestMatchIoU = iou
                bestMatchBbox = [dx, dy, dw, dh]
                dtstr = str(item['category_id'])

    return bestMatchIoU, bestMatchBbox, dtstr


## --------------------------------------------------------------------
#   draw boxes on image
## --------------------------------------------------------------------

def draw_boxes(image, data, type_of_det, size_of_gt, conf_thr) : 

    # setup contexts and stats
    gts = data['Gts']
    dts = data['Dts']
    numDts = data['numDTs']
    numTps = data['numTPs']
    numFps = data['numFPs']
    numGts = data['numGTs']
    numFns = data['numFNs']
    fileName = data['file_name']
    skippedTps = 0
    skippedFps = 0
    skippedFns = 0
    skippedGts = 0

    # keep count of number of boxes drawn on the image
    numBoxesDrawn = 0

    type_of_det = type_of_det.lower()
    size_of_gt = size_of_gt.lower()
    print 'type_of_det {}, size_of_gt {}, conf_thr {}'.format(type_of_det, size_of_gt, conf_thr)

    for item in dts : 

        a,b,c,d = item['bbox']
        x = int(a)
        y = int(b)
        w = int(c)
        h = int(d)

        dtstr = str(item['category_id'])
        dtscore = item['score']

        ## filter based on requested confidence threshold cutoff
        if dtscore < conf_thr :
            if item['isFP'] is True :
                skippedFps += 1
            else : 
                skippedTps += 1
            continue

        ## filter based on size 
        dtsize = item['size']
        if dtsize == 'all' : 
            area = w*h
            if area <= 32*32 : 
                dtsize = 'small'
            elif area > 96*96 : 
                dtsize = 'large'
            else : 
                dtsize = 'medium'

        if size_of_gt != 'all' and dtsize != size_of_gt : 
            if item['isFP'] is True :
                skippedFps += 1
            else : 
                skippedTps += 1
            continue

        if item['isFP'] is True :
            if type_of_det == 'all' or type_of_det == 'fp': 
                cv2.putText(image, (dtstr), (x, y-5), 0, 0.3, (225,225,0))
                numBoxesDrawn += 1
                if dtscore > 0.2 : 
                    cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,0), 1)
                else : 
                    cv2.putText(image, '*', (x, y), 0, 0.3, (225,225,0))
            else : 
                skippedFps += 1
        else : 
            if type_of_det == 'all' or type_of_det == 'tp': 
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.putText(image, (dtstr), (x, y-5), 0, 0.3, (0,225,225))
                numBoxesDrawn += 1
            else : 
                skippedTps += 1

    for item in gts : 

        a,b,c,d = item['bbox']
        x = int(a)
        y = int(b)
        w = int(c)
        h = int(d)

        #gtsize = 'all'
        #area = w*h
        #if area <= 32*32 : 
        #    gtsize = 'small'
        #elif area > 96*96 : 
        #    gtsize = 'large'
        #else : 
        #    gtsize = 'medium'

        ## filter based on requested size_of_gt
        gtsize = item['size']
        if size_of_gt != 'all' and gtsize != size_of_gt : 
            if item['isFN'] is True : 
                skippedFns += 1
            else : 
                skippedGts += 1
            continue

        ## filter based on requested type_of_det
        gtstr = str(item['category_id'])
        if item['isFN'] is True : 
            if type_of_det == 'all' or type_of_det == 'fn' or type_of_det == 'fnfp': 
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
                cv2.putText(image, (gtstr), (x, (y+h+5)), 0, 0.3, (225,0,255))
                numBoxesDrawn += 1
                if type_of_det == 'fnfp' : 
                    found, fp_bbox, dtstr = findOverlappedBbox(item['bbox'], dts, type_of_det='fp')
                    if found > 0 : 
                        fp_x, fp_y, fp_w, fp_h = fp_bbox
                        cv2.rectangle(image, (fp_x,fp_y), (fp_x+fp_w, fp_y+fp_h), (255,255,0), 1)
                        cv2.putText(image, (dtstr), (fp_x, fp_y-5), 0, 0.3, (225,225,0))
                        numBoxesDrawn += 1

            else : 
                skippedFns += 1
        else : 
            if type_of_det == 'all' or (type_of_det != 'fn' and type_of_det != 'fnfp'): 
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
                cv2.putText(image, (gtstr), (x, (y+h+5)), 0, 0.3, (225,0,255))
                numBoxesDrawn += 1
            else : 
                skippedGts += 1


    cv2.putText(image, ('GT Color'), (10,20), 0, 0.3, (0,255,0))
    cv2.putText(image, ('TP Color'), (90,20), 0, 0.3, (0,0,255))
    cv2.putText(image, ('FP Color'), (170,20), 0, 0.3, (255,255,0))
    cv2.putText(image, ('FN Color'), (240,20), 0, 0.3, (255,0,0))
    cv2.putText(image, (str(fileName)), (10,40), 0, 0.3, (255,255,255))

    countString = 'orig Number of GTs: {}, DTs {}, TPs {}, FNs {}, FPs {}'.format(numGts, numDts, numTps, numFns, numFps)
    #print countString
    skippedDts = skippedTps + skippedFps
    skippedGts = skippedGts + skippedFns
    skippedString = 'skip Number of GTs: {}, DTs {}, TPs {}, FNs {}, FPs {}'.format(skippedGts, skippedDts, skippedTps, skippedFns, skippedFps)

    cv2.putText(image, (str(countString)), (10,60), 0, 0.4, (255,255,255))
    cv2.putText(image, (str(skippedString)), (10,80), 0, 0.4, (255,255,255))

    return image, numBoxesDrawn

## --------------------------------------------------------------------
#   handle json data file
## --------------------------------------------------------------------

def display_json_data(srcFile, compareFile, sort_column, saveDir, showImages, type_of_det, size_of_gt, conf_thr) : 

    fields = ['file_name', 'id', 'numGTs', 'numTPs', 'numFPs', 'numFNs']
    compare = False

    if not compareFile is None : 
        cfile_input = open(compareFile, 'r')
        cfile_data = json.load(cfile_input)
        #fields = cfile_data.fieldnames
        #cmp_data = sorted(cfile_data, key=lambda row: (row[fields[sort_column]], row[fields[0]]))
        cmp_data = sorted(cfile_data, key=lambda row: (row[fields[1]], row[fields[0]]))
        compare = True

    with open(srcFile, 'r') as f_input:
        src_input = json.load(f_input)

        # if sort has same values, preserves rows using field 0, which is the image_file_name
        reverse = False
        if sort_column > 1 : 
            reverse = True
        print 'sort_column is {} and reverse is {}'.format(fields[sort_column], reverse)
        src_data = sorted(src_input, key=lambda row: (row[fields[sort_column]], row[fields[0]]),reverse=reverse)
    
    # use openCV and start dispalying these files in the sorted order
    for row in src_data :
        image_file_name = row['file_name']
        image_id = row['id']

        win_name = 'viewDets : Image Id {}'.format(image_id)
        image = cv2.imread(image_file_name)
        src_image_with_boxes, gotBoxesSrc = draw_boxes(image, row, type_of_det, size_of_gt, conf_thr)
        image_to_show = src_image_with_boxes

        if compare is True : 
            #cmp_row = [c_row for c_row in cmp_data if c_row['id'] == image_id][0]
            rec2cmp = [c_row for c_row in cmp_data if c_row['id'] == image_id]
            if rec2cmp : 
                cmp_row = rec2cmp[0]
                #print 'image_id is {} and cmp_row is {}'.format(image_id, cmp_row)
                cmp_image_file_name = cmp_row['file_name']
                cimage = cv2.imread(cmp_image_file_name)
                cmp_image_with_boxes, gotBoxesCmp = draw_boxes(cimage, cmp_row, type_of_det, size_of_gt, conf_thr)
            else :
                cmp_image_with_boxes = np.random.uniform(size=(image_to_show.shape)) + 100.0


            ## stack the two images
            np_stack_h = np.hstack((src_image_with_boxes, cmp_image_with_boxes))
            np_concat_h = np.concatenate((src_image_with_boxes, cmp_image_with_boxes), axis=1)
            image_to_show = np_concat_h

        # save the image if requested into the dir and using the src_file_name with a prefix "cmp_and_stacked" 
        if not saveDir is None and gotBoxesSrc > 0: 
            file2save = saveDir + '/' + os.path.basename(image_file_name) 
            #print 'file2save is: {}'.format(file2save)
            cv2.imwrite(file2save, image_to_show)

        # move and wait - make this as an option i.e., to view these images
        if showImages is True and gotBoxesSrc > 0: 
            cv2.imshow(win_name,image_to_show)
            cv2.moveWindow(win_name, 100, 100)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
            cv2.destroyAllWindows()

    print 'json: DONE!'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--src_file", type=str, help="per image detection evaluation file, created when running evaluation scripts. Either the .csv file or .json file")

    parser.add_argument("--sort_field", type=int, default=1, choices=range(1,6), help="The options for sort fields are: 1: image_id, 2: GTs, 3: TPs, 4: FPs, 5: FNs")

    parser.add_argument("--compare_file", type=str, help="images to display for a side-by-side view against src file. This file is a per image detection evaluation file, created when running evaluation scripts. Either the .csv file or .json file")

    parser.add_argument("--save_dir", type=str, help="Absolute path to a directory to save the images with bounding boxes, when using .json data src files")

    parser.add_argument("--show_images", action='store_true', help="To display images or not. Only useful with json input files.")

    parser.add_argument("--dt_type", type=str, default='all', help="Type of detection to display. Options to choose from: all*, TP, FP, FN, FNFP. Only useful with json input file")

    parser.add_argument("--gt_size", type=str, default='all', help="Size of annotations to display. Options to choose from: all*, small, medium, large. Only useful with json input file")

    parser.add_argument("--conf_threshold", type=float, default=0.01, help="confidence threshold of detections to use for display. Options to choose from: {0.01*|[0.01 to 0.9]}. Only useful with json input file")

    #parser.add_argument("--fid_source", action='store_true', help="sort based on fid scores & show images with FID scores smallest to largest")
    parser.add_argument("--fid_source", type=str, default=None, help="batch|single: sort based on fid scores, save as batch or single file")

    parser.add_argument("--num_fid_images_to_save", type=int, default=1000, help="Number of images to save based on the best FID scores")

    args = parser.parse_args()
    srcFile = args.src_file or None
    sort_column = args.sort_field or 1
    compareFile = args.compare_file or None
    saveDir = args.save_dir or None
    showImages = args.show_images or False
    type_of_det = args.dt_type or 'all'
    size_of_gt = args.gt_size or 'all'
    conf_thr = args.conf_threshold or 0.01
    fidSource = args.fid_source 
    numFidImages2Save = args.num_fid_images_to_save 

    compare = False
    cext = '.junk'
    fext = '.junk'

    if not os.path.exists(srcFile) : 
        print 'src_file {} does NOT exists. Bailing out'.format(srcFile)
        sys.exit('src_file {} does NOT exists. Bailing out'.format(srcFile))
    else :
        fname,fext = os.path.splitext(srcFile)
        #print 'src file extension is {}'.format(fext)
    
    if compareFile is None : 
        print 'FYI: compare_file input is not proved'
    else :
        if not os.path.exists(compareFile) : 
            sys.exit('compare_file {} does NOT exists. Bailing out'.format(compareFile))
        cfname, cext = os.path.splitext(compareFile)

        # bailout if the src and compare files are NOT of the same type
        if fext != cext : 
            print('src_file and compare_file are NOT of the same type {}:{}. Bailing out'.format(fext, cext))
            sys.exit('src_file and compare_file are NOT of the same type {}:{}. Bailing out'.format(fext, cext))


    if not saveDir is None :
        if os.path.exists(saveDir) : 
            print 'clearing save_dir {}'.format(saveDir)
            shutil.rmtree(saveDir)
        print 'creating save_dir {}'.format(saveDir)
        os.makedirs(saveDir)

    if fidSource != None : 
        if fidSource == 'batch' : 
            display_fid_batch_data(srcFile, saveDir, numFidImages2Save, showImages)
        else :
            display_fid_data(srcFile, saveDir, numFidImages2Save, showImages)
    elif fext == ".json" : 
        display_json_data(srcFile, compareFile, sort_column, saveDir, showImages, type_of_det, size_of_gt, conf_thr)
    else :
        display_csv_data(srcFile, compareFile, sort_column)

    print 'DONE!'

################################################################################

'''
Q4-2018 kedar

Q1-2019 kedar : 
                Enhanced with filtering options to be able to dissect detections
                dt_type, gt_size, conf_threshold options are added

Q1-2019 kedar : 
                added option to view images from FID output file

As part of evaluation scripts, when you save the images with detections, it will
also create a csv file, where each row has the following values:
    ['image_file_name', image_id, 'GTs', 'TPs', 'FPs', 'FNs']

By default: even when images are not saved, a 'json' file is created in the same directory as
    where the detections.txt file is available. This json file will have details the same
    as the csv file and ALSO has the 'Gt' and 'Dt' bboxes, with meta data
    indicating whether the detections are TP/FP and GT is FN or not. These elements are part of each
    image record (same as in GT file).This json file can also be used with this script. 
    It would generate the same format of vieweing images as with the csv file.

NOTE: detections.txt is generated by testScript.sh in the 'workspace' directory. In the case of 
    evaluateDetections.sh, 'detections.txt' is provided as input. 

NOTE: both the src_file and compare_file MUST be of the same kind

This script is a tool to view detections on images from such a src file, where
the input determines what images to look into i.e., sorted list with TPs or FPs or FNs, 
from most to least.  Alternatively, it can take two files, where sorting is based on the 
first one and then displays the corresponding images from both, to look into detections on 
the two images side-by-side.

The inputs are: 
        --src_file 
        --sort_filed : 1: image_id, 2: GTs, 3: TPs, 4: FPs, 5: FNs
        --compare_file
        following input arguments are only used with json input files
        --show_images 
        --save_dir
        --dt_type : tp, fp, fn, fnfp, all(default)
                    fnfp: will display FNs and FPs also
        --gt_size : small, medium, large, all(default)
        --conf_threshold: only detections higher than the threshold are displayed (0.01 default)
        --fid_source: false (default) or true. Must be specified when the src_file is output from FID scoring scripts
        --num_fid_images_to_save: number of best scoring fid images to save

NOTE: This script uses the filenames in the input file(s) as is. If you run the evaluation scripts
with save images option, the filenames in csv files and the location of the saved images
match and this script only works like this currently. In the case of json, 'file_name" element is 
used as the file to open - which is the filename with absolute path in the testdata_set/test/Data directory

In the case of "fid_source" option, the records in the file are sorted based on FID scores and corresponding
        images are showed in a stacked fashion. 

Typically, the src files must be from evaluating against the SAME TEST DATASET for a typical use-case

e.g., usage: 

#python viewDetections.py --src_file=/home/kmadineni/junk/test_catId_3_dot0_tp_fp_fn_boxes-bup.json --show_images --gt_size=small --dt_type=fn --conf_threshold=0.2 --compare_file=/home/kmadineni/junk/test_catId_3_dot0_tp_fp_fn_boxes-bup.json

#python viewDetections.py --src_file /home/kmadineni/junk/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_rel_adas.json --sort_field 5 --compare_file=/home/kmadineni/work/crepo/totcaffe/models/refinedet/jan25_2019_inter_rgb_s3/workspace/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_iter_750.json --save_dir /tmp/kmadineni/vis_tool_images_jan29_meeting/large_fn_adas_rel_vs_synth_best --dt_type fnfp --gt_size large

#python viewDetections.py --src_file /home/kmadineni/junk/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_rel_adas.json --sort_field 5 --compare_file=/home/kmadineni/work/crepo/totcaffe/models/refinedet/jan25_2019_inter_rgb_s3/workspace/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_iter_750.json --save_dir /tmp/kmadineni/vis_tool_images_jan29_meeting/large_fn_adas_rel_vs_synth_best --dt_type fnfp --gt_size large --show_images

#python viewDetections.py --src_file /home/kmadineni/junk/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_rel_adas.json --sort_field 5 --compare_file=/home/kmadineni/work/crepo/totcaffe/models/refinedet/jan25_2019_inter_rgb_s3/workspace/t07_jul12_test_detectionResults_catId_3_dot0_tp_fp_fn_boxes_iter_750.json --save_dir /tmp/kmadineni/vis_tool_images_jan29_meeting/large_fn_adas_rel_vs_synth_best --dt_type fn --gt_size large --show_images

#python viewDetections.py --src_file=/mnt/data/flir-data/syncity/intersection/jan25_2019/fid_miami_t07.json --fid_source --save_dir=/mnt/data/flir-data/syncity/intersection/jan25_2019/fid_miami

'''


