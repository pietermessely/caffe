"""
@Date:   Q1'2018
@author: kedarM
@Notes:  Mods, Augments & Generalization from existing script(s)
"""

import json
import cv2 
import matplotlib.pyplot as plt

import os
import argparse

def find(lst, key1, value1, key2, value2):
	grp = []	
	for i, dic in enumerate(lst):
		if (dic[key1] == value1) and (dic[key2] == value2) :
			grp.append(i)
	return grp

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

def find_DT(lst, key1, value1, key2, value2, key3, value3_min, value3_max):
	grp = []	
	for i, dic in enumerate(lst):
                # if requested value of catId (value2) is '0' indicating 'all' : ignore key2/value2 check
                if (value2 == 0) : 
		    if (dic[key1] == value1) and (value3_min <= dic[key3][2]*dic[key3][3] <= value3_max):
			grp.append(i)
                else :
		    if (dic[key1] == value1) and (dic[key2] == value2) and (value3_min <= dic[key3][2]*dic[key3][3] <= value3_max):
			grp.append(i)
	return grp

def intersection_area(R1, R2):
	x = max(R1[0], R2[0])
	y = max(R1[1], R2[1])
	w = min(R1[0]+R1[2], R2[0]+R2[2]) - x
	h = min(R1[1]+R1[3], R2[1]+R2[3]) - y
	if w<0 or h<0: return 0
	return w*h

bbox_size = 'all'

if bbox_size == 'small':
	area_min = 9; area_max = 1024
elif bbox_size == 'medium':
	area_min = 1024; area_max = 9216
elif bbox_size == 'large':
	area_min = 9216; area_max = 327680
elif bbox_size == 'all':
	area_min = 9; area_max = 327680

## gather environmental variables : Assume these exist as it will be called from another script
modelsRoot = os.environ.get("FLIR_MODELS_ROOT")
dataRoot = os.environ.get("ML_TEST_DATA_SET_ROOT")
caffeRoot = os.environ.get("CAFFE_ROOT")

# input arguments 
parser = argparse.ArgumentParser()
parser.add_argument("datasetDir", type=str,
                    help="Source Directory of Data Set to be used. Full Path should be provided")
parser.add_argument("gtFile", type=str,
                    help="Ground Truth JSON file. Full Path should be provided")
parser.add_argument("detResultsFilePrefix", type=str,
                    help="Detection Results JSON file prefix. Full Path should be provided")
parser.add_argument("catId", type=int, choices=range(0,90),
                    help="results for requested category")
#parser.add_argument("modelFileDir", type=str,
#                    help="Model File Dir. Relative Path from FLIR_MODELS_ROOT")
#parser.add_argument("confThrBegin", type=float, choices=(0.3,0.4),
#                    help="Confidence Threshold starting value .. ending will always be 0.9 ")

args = parser.parse_args()
datasetDir = str(args.datasetDir)
annJson = str(args.gtFile)
detResultsFilePrefix = str(args.detResultsFilePrefix)
catId = int(args.catId)
#modelFileDir = str(args.modelFileDir)
#confTheBegin = float(args.confThrBegin)

#datasetDir = '%s/%s/test'%(dataRoot,dataSrcDir)
#annJson = '%s'%(gtFile)
imgDir = '%s/Data'%(datasetDir)

#modelsDir = '%s/%s'%(modelsRoot,modelFileDir)
#detResultsFileTmp = '%s/%s'%(modelsDir,detResultsFilePrefix)

#catName = ['background', 'person', 'bike', 'car', 'dog']
#iteration='4cat_50K'
#baseDir = '/mnt/data/FR3-2/'
#gt_json = 'FR3-2_GT.json'
#imgDir = '%s/merge_test/Data/'%(baseDir)
#annJson = '%s/GT_DT_json/%s'%(baseDir, gt_json)

print 'Plotting RoC: \n\tcatId %s \n\t datasetDir %s \n\t annJson %s \n\t detectionResultsPrefix %s \n'%(catId, datasetDir, annJson, detResultsFilePrefix)
ann = open(annJson).read()
Gt = json.loads(ann)
Gt = Gt['annotations']

FA = []
PD = []
fig = plt.figure(1)

confThrList = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
#confThrList = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  

#for i in range(0,7):
for i in range(0,7):

    thrd = confThrList[i]
    thrStr = 'dot%s'%(str(int(thrd*10)))
    resJson_file = '%s_%s.json'%(detResultsFilePrefix,thrStr)

    res = open(resJson_file).read()
    Dt = json.loads(res)

    imgIds = []

    for i in range(len(Dt)):
        imgIds.append(Dt[i]['image_id'])

    imgIds = list(set(imgIds))
    imgIds.sort() 


    total_num_dt = 0.0
    total_num_gt = 0.0
    chosen_num_dt = 0.0

    FA_per_img = []
    PD_per_img = []

    for imgId in imgIds:

	index_dt = find_DT(Dt, 'image_id', imgId, 'category_id', catId, 'bbox', area_min, area_max)
	index_gt = find_GT(Gt, 'image_id', imgId, 'category_id', catId, 'area', area_min, area_max)

	false_per_img = 0
	intersec_per_img = 0

	if (len(index_dt) != 0) and (len(index_gt) != 0):
	    areaGts = []
            total_num_dt = total_num_dt + len(index_dt)
            total_num_gt = total_num_gt + len(index_gt)

	    for ind_gt in index_gt:
		areaGts.append(Gt[ind_gt]['area'])

    	    areaGt_per_img = sum(areaGts)
	    areaDt_per_img = 0

	    for ind_dt in index_dt:
	        Rd = Dt[ind_dt]['bbox']
	        areaDt = int(Rd[2]*Rd[3])
	        areaDt_per_img = areaDt_per_img + areaDt

	        IoU_dt = 0
	        ind_dt_gt = 0
	        intersec_dt = 0

	        for ind_gt in index_gt:
		    Rg = Gt[ind_gt]['bbox']
		    intersec = intersection_area(Rg, Rd)
		    IoU = intersec/(areaGts[index_gt.index(ind_gt)] + areaDt - intersec)

		    if IoU > IoU_dt:
		        IoU_dt = IoU
		        ind_dt_gt = ind_gt
		        intersec_dt = intersec

		false_dt = areaDt - intersec_dt				
	        false_per_img = false_per_img + false_dt				
	        intersec_per_img = intersec_per_img + intersec_dt

	    FA_per_img.append(false_per_img/areaDt_per_img)
	    PD_per_img.append(intersec_per_img/areaGt_per_img)

        elif (len(index_dt) != 0) and (len(index_gt) == 0):
	    areaDt_per_img = 0
	    false_per_img = 0
	    for ind_dt in index_dt:
		Rd = Dt[ind_dt]['bbox']
		areaDt = int(Rd[2]*Rd[3])
		areaDt_per_img = areaDt_per_img + areaDt
		false_dt = areaDt
		false_per_img = false_per_img + false_dt

	    FA_per_img.append(false_per_img/areaDt_per_img)

        elif (len(index_dt) == 0) and (len(index_gt) != 0):
	    FA_per_img.append(0.0)
	    PD_per_img.append(0.0)


    if len(FA_per_img)>0 and len(PD_per_img)>0:
	FA_per_thrd = sum(FA_per_img)/len(FA_per_img)
	PD_per_thrd = sum(PD_per_img)/len(PD_per_img)
	FA.append(FA_per_thrd)
	PD.append(PD_per_thrd)

	print('(FP (False Alarm), PD (Detection Probablity)) for confidence threshold %s = (%.3f, %.3f)'%(thrd, FA_per_thrd, PD_per_thrd))

    else:
	print('(FA, PD) N/A for confidence threshold %s'%(thrd))

# plot the curve
plt.plot(FA, PD, label=str(catId))
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True)

titleStr = 'ROC Curve : Category %s'%(str(catId))
fig.suptitle(str(titleStr))

plt.xlabel('false alarm')
plt.ylabel('detection probability')
#plt.ylim([0.0, 1.0])
#plt.xlim([0.0, 1.0])
plt.show()

