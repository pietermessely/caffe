"""
@Date:   Q4'2018
@author: kedarM
@Notes:  revamped initial implementation from Q1'2018, based on other changes
            CAUTION: This code is called ONLY when catId is > 0 i.e., specific
                        category. It takes about 46s to calculate for each IoU
                        threshold, with about 100K detections. With 10 such
                        thresholds, it takes about 8 minutes roughly. 
            TODO: 
                    (1) use numpy arrays in find_GT and find_DT, streamline some logic
                    (2) save plots
                    (3) save the precision/recall values into a csv file
"""

import json
import cv2 
import matplotlib.pyplot as plt
import os
import argparse
import time
import numpy as np

##
def find(lst, key1, value1, key2, value2):
    grp = []	
    for i, dic in enumerate(lst):
        if (dic[key1] == value1) and (dic[key2] == value2) :
            grp.append(i)
    return grp

##
# find_GT(Gt, 'image_id', imgId, 'category_id', catId, 'area', area_min, area_max)
##
def find_GT(lst, key1, value1, key2, value2, key3, value3_min, value3_max):
    grp = []	
    for i, dic in enumerate(lst): 
        # if requested value of catId (value2) is '0' indicating 'all' : ignore key2/value2 check
        if (value2 == 0) :
            if (dic[key1] == value1) and (value3_min <= dic[key3] <= value3_max):
                grp.append(i)
        else :
            if (dic[key1] == value1) and (dic[key2] == value2) and (value3_min <= dic[key3] <= value3_max):
                grp.append(i)
    return grp

##
##find_DT(Dt, 'image_id', imgId, 'category_id', catId, 'bbox', area_min, area_max, 'score', confThr)
def find_DT(lst, key1, value1, key2, value2, key3, value3_min, value3_max, key4, value4):
    grp = []	
    for i, dic in enumerate(lst):
        # if requested value of catId (value2) is '0' indicating 'all' : ignore key2/value2 check
        if (value2 == 0) :
            if (dic[key1] == value1) and (value3_min <= dic[key3][2]*dic[key3][3] <= value3_max) and (value4 <= dic[key4]):
                grp.append(i)
        else :
            if (dic[key1] == value1) and (dic[key2] == value2) and (value3_min <= dic[key3][2]*dic[key3][3] <= value3_max) and (value4 <= dic[key4]):
                grp.append(i)
    return grp

##
def intersection_area(R1, R2):
    x = max(R1[0], R2[0])
    y = max(R1[1], R2[1])
    w = min(R1[0]+R1[2], R2[0]+R2[2]) - x
    h = min(R1[1]+R1[3], R2[1]+R2[3]) - y

    if w<0 or h<0: 
        return 0

    return w*h

##
def get_area_min_max(bbox_size) :
    # setup area range 
    if bbox_size == 'small':
	area_min = 9; area_max = 1024               ## 3*3 to 32*32 in pixels
    elif bbox_size == 'medium':
	area_min = 1024; area_max = 9216            ## 32*32 to 96*96 in pixels
    elif bbox_size == 'large':
	area_min = 9216; area_max = 99999999        ## 96*96 to some arbitrary large value
    else :
	area_min = 9; area_max = 99999999           ## defaults to all sizes 

    return area_min, area_max

##
#   calculate recall and precision values at the passed IoU for all values in confThr range
##

def calculate_pr_values(Gt, Dt, imgIds, catId, bbox_area, iouThreshold): 

    confThrList = np.linspace(0.01, 0.9, 10)
    #print 'confThrList is {}'.format(confThrList)

    # setup area_min and area_max values based on passed area min and max value
    area_min, area_max = get_area_min_max(bbox_area)
    #print 'area min:max is {}:{}'.format(area_min,area_max)

    # setup precision and recall lists
    plist = []
    rlist = []

    # prepend with max precision value for the curve to look nicer
    plist.append(1.0)
    rlist.append(0.0)
    first = True

    #print 'Calculating P-R values @ iouThr {}, catId {}'.format(iouThreshold, catId)

    confThrList = confThrList[::-1]

    for confThr in confThrList :
        total_num_dt = 0.0
        total_num_gt = 0.0
        chosen_num_dt = 0.0
        precision = 0.
        recall = 0.

        #print 'generating pr values for confThr {}'.format(confThr)

        for imgId in imgIds:

            #print 'checking image id: {}'.format(imgId)
	    index_dt = find_DT(Dt, 'image_id', imgId, 'category_id', catId, \
                    'bbox', area_min, area_max, 'score', confThr)
	    index_gt = find_GT(Gt, 'image_id', imgId, 'category_id', catId, \
                    'area', area_min, area_max)

	    if (len(index_dt) != 0) and (len(index_gt) != 0):
                total_num_dt = total_num_dt + len(index_dt)
                total_num_gt = total_num_gt + len(index_gt)

	        for ind_dt in index_dt:
	            det_box = Dt[ind_dt]['bbox']
	            areaDt = det_box[2]*det_box[3]

	            for ind_gt in index_gt:
		        gt_box = Gt[ind_gt]['bbox']
                        areaGt = gt_box[2]*gt_box[3]
		        intersec = intersection_area(gt_box, det_box)
		        IoU = intersec/(areaGt + areaDt - intersec)

		        if IoU >= iouThreshold:
                            chosen_num_dt = chosen_num_dt + 1
                            break

                    index_gt = [x for x in index_gt if x != ind_gt]
            else : 
                total_num_dt = total_num_dt + len(index_dt)
                total_num_gt = total_num_gt + len(index_gt)

        # save the precision and recall for each confThr value
        if total_num_gt != 0 : 
            recall = chosen_num_dt/total_num_gt

            try : 
                precision = chosen_num_dt/total_num_dt
            except ZeroDivisionError :
                precision = 0.

            rlist.append(recall)
            plist.append(precision)

            # setup value of precision at recall 0 
            if first is True:
                first = False
                plist[0] = precision

        #print "TPs: %d, NumGt: %d, NumDt(TP+FP): %d" %(chosen_num_dt, total_num_gt, total_num_dt)
        #print 'confThr: %.3f precision: %.3f recall: %.3f'%(confThr, precision, recall)

    # return the lists for the passed IoU
    # appending with max recall value for the curve to look nicer
    plist.append(0.0)
    rlist.append(rlist[-1])
    return plist, rlist

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

##
if __name__ ==  "__main__" :

    ## gather environmental variables : Assume these exist as it will be called from another script
    modelsRoot = os.environ.get("FLIR_MODELS_ROOT")
    dataRoot = os.environ.get("ML_TEST_DATA_SET_ROOT")
    caffeRoot = os.environ.get("CAFFE_ROOT")

    # input arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetDir", type=str, \
            help="Source Directory of Data Set to be used. Full Path should be provided")
    parser.add_argument("gtFile", type=str,\
            help="Ground Truth JSON file. Full Path should be provided")
    parser.add_argument("detResultsFile", type=str,\
            help="Detection Results JSON file. Full Path should be provided")
    parser.add_argument("catId", type=int, choices=range(0,90),\
            help="results for requested category")

    args = parser.parse_args()
    datasetDir = str(args.datasetDir)
    gt_file = str(args.gtFile)
    det_file = str(args.detResultsFile)
    catId = int(args.catId)
    imgDir = '%s/Data'%(datasetDir)

    #  default bbox size
    bboxsize = 'all'

    start_time = time.time()

    print 'Plotting PR Curve: catId %s gt_file %s det_file %s'%(catId, gt_file, det_file)
    precisions_list = []
    recalls_list = []
    fig = plt.figure(figsize=(10,8))
    plt.gca()

    ann = open(gt_file).read()
    GtAll = json.loads(ann)
    Gt = GtAll['annotations']

    # read detection resutls file and setup some contexts
    res = open(det_file).read()
    Dt = json.loads(res)

    # gather list of images from ground truth file
    imgIds = []
    for i in range(len(Gt)):
        imgIds.append(Gt[i]['image_id'])
    imgIds = list(set(imgIds))
    imgIds.sort() 
    #print 'image Ids are {}'.format(imgIds)

    # gather precision and recall for different confidence thresholds at each IoUs 
    #       i.e., PR curve at each IoU
    idx = 0
    #for iouThr in [0.5] : 
    for idx, iouThr in enumerate(np.linspace(0.5, 0.95, 10)) : 
        ltime = time.time()
        precisions_list, recalls_list = calculate_pr_values(Gt, Dt, imgIds, catId, bboxsize, iouThr)
        print 'to calculate PR for confThr {} took {:.4f} secs'.format(iouThr, (time.time() - ltime))
        # plot the PR curve at this IoU
        plt.plot(recalls_list, precisions_list, label='{:.2f}'.format(iouThr), color=COLORS[idx])

        ## other way
        #precisions = np.array(precisions_list)
        #recalls = np.array(recalls_list)
        #prec_at_rec = []
        #recall_levels_list = np.linspace(0.0, 1.0, 11)
        #for recall_level in np.linspace(0.0, 1.0, 11):
        #    try:
        #        args = np.argwhere(recalls >= recall_level).flatten()
        #        prec = max(precisions[args])
        #    except ValueError:
        #        prec = 0.0
        #    prec_at_rec.append(prec)
        #plt.plot(recall_levels_list, prec_at_rec, label='prWithRcLevel', color='r-')

    titleStr = 'PR Curve : Category ID %s'%(str(catId))
    fig.suptitle(str(titleStr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.2])
    plt.xlim([0.0, 1.3])
    plt.legend(loc='upper right', title='IoU Threshold', frameon=True)

    # vertical dashed lines
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')

    end_time = time.time()
    print 'Generating PR Plot for Category Id {} took {:.4f} secs'.format(catId, (end_time - start_time))

    plt.show()

