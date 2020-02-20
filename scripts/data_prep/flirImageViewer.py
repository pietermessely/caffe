"""
Q4 2017         Class for viewing images
@author         kedar m

"""
__author__ = 'kedarm'

import os
import numpy as np
import json
import cv2
import copy
import csv
import pwd

class flirViewer:

    def __init__(self, resJson=None, annJson=None, scoreThr=0.0, testCatId=None, detectionImagesOutputDir=None, resize_height=0, resize_width=0):

        if not (resJson is None and annJson is None):
            self.resJson = resJson
            self.annJson = annJson

        else:
            print 'ERROR: Input Arguments!\n'
            print 'GT Annotation and Detection Results Files are NOT provided\n'
            self.resJson = None
            self.annJson = None

        self.scoreThr = scoreThr
        self.testCatId = testCatId

        self.dtImageIDs = []
        self.gtImageIDs = []
        self.fnImageIDs = []
        self.fpImageIDs = []

        self.Gt = None
        self.GtAnnotations = None
        self.GtAnnotations_copy = None
        self.GtImages = None

        self.Dt = None
        self.Dt_copy = None

        self.fnegFile = None
        self.fposFile = None
        
        self.annLoaded = [ None, None ]
        
        self.img2dtidx = {}
        self.img2gtidx = {}
        self.img2gtimidx = {}
        self.rangemax = None
        
        self.loadAssets()

	if (detectionImagesOutputDir is None) or (detectionImagesOutputDir == "None"):
            self.detectionImagesOutputDir = None
        else:
            self.detectionImagesOutputDir = detectionImagesOutputDir

        self.resize_height = resize_height
        self.resize_width = resize_width

    def _find(self, lst, key, value):
        grp = []
        for i, dic in enumerate(lst):
            if dic[key] == value:
                grp.append(i)
        return grp


    def viewDetections(self, numOfImages=-1):
        '''
            Setup GT Annotations & DT bounding boxes on the image and
            call the image viewer
        '''


        if (self.resJson is None or self.annJson is None):
            print 'DetResultsFile: {}'.format(self.resJson)
            print 'GT-AnnotationsFile: {}'.format(self.annJson)
            print 'Need to Initialize with above files'
            raise Exception('File NOT provided')
            #return 'File NOT provided'

        # setup the window & size it or use defaults
        #cv2.namedWindow ('flirViewer', cv2.CV_WINDOW_NORMAL)
        #cv2.resizeWindow ('flirViewer', 512, 512)
        #cv2.namedWindow ('flirViewer', cv2.CV_WINDOW_AUTOSIZE)
        #cv2.namedWindow ('flirViewer')

        resize_width = self.resize_width
        resize_height = self.resize_height

        # setup test category Id  & scoreThreshold
        testCatId = self.testCatId
        scoreThr = self.scoreThr
        
        (Gt, GtAnnotations, GtImages, gtImageIds, Dt, dtImageIds, rangemax, img2dtidx, img2gtidx, img2gtimidx) = self.loadAssets()
        imgIds = dtImageIds
        if numOfImages > -1: rangemax = min(numOfImages, rangemax)
        for jj in range(rangemax):
            if (jj < 0):        # dirty debug
                imgId = 32	# 1763 1772 1745 1746 1749
            else :
                imgId = imgIds[jj]

            # index_dt = self._find(Dt, 'image_id', imgId)
            # index_gt = self._find(GtAnns, 'image_id', imgId)
            try: index_dt = img2dtidx[imgId] # self._find(Dt, 'image_id', imgId)  
            except KeyError: index_dt = []
            try: index_gt = img2gtidx[imgId] # self._find(GtAnns, 'image_id', imgId)
            except KeyError: index_gt = []

            # (km) get the image file name from GT file
            # index_fname = self._find(GtImgs, 'id', imgId)
            try: index_fname = img2gtimidx[imgId]
            except KeyError: index_fname = []
            #print 'index_fname : {}'.format(index_fname) + '\n'
            dic = GtImgs[int(index_fname[0])]
            fname = dic['file_name']
            #print 'image file name is: ' + fname + '\n'

            ## ==================================================
            ##      resize for map-prime case
            ## ==================================================
            if resize_height is not 0 and resize_width is not 0 :
                image = cv2.imread(fname)
                frame = cv2.resize(image, (resize_height, resize_width))
            else :
                frame = cv2.imread(fname)

            # get the detection bbox information
	    box_Dt = np.zeros((len(index_dt), 4))
	    c=0
	    for ind in index_dt:
                dic = Dt[ind]
                catId = dic['category_id']
                score = dic['score']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    continue

                # only show images if score is >= scoreThr
                if( (score < scoreThr) ) :
                    continue

                box_Dt[c,:] = dic['bbox']
                x = int(box_Dt[c,0])
                y = int(box_Dt[c,1])
                w = int(box_Dt[c,2])
                h = int(box_Dt[c,3])

                area = int(w*h)
                if area <= 32*32 :
                    astring = 'S'
                elif area > 96*96 :
                    astring = 'L'
                else :
                    astring = 'M'

                # setup the txt to be displayed on the image
                displayStr = str(catId) + ", " + str(score) + ", " + str(astring)

                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)
                cv2.putText(frame, str(displayStr), (x, y-5), 0, 0.5, (0,255,255))
                c = c + 1

            # get the GT bbox information
            box_Gt = np.zeros((len(index_gt), 4))
            c=0
            for ind in index_gt:
                dic = GtAnnotations[ind]
                catId = dic['category_id']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    continue

                box_Gt[c,:] = dic['bbox']
                x = int(box_Gt[c,0])
                y = int(box_Gt[c,1])
                w = int(box_Gt[c,2])
                h = int(box_Gt[c,3])
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),1)
                # TODO: post name instead of integer
                cv2.putText(frame, str(catId), (x, y-5), 0, 0.5, (0,255,0))
                c = c + 1

            # (km) : Show the image filename also.
            #	 Input args for putText:
            #	    image, text, coordinates_for_start_of_text, font_type, font_scale, font_color
            #
            cv2.putText(frame, ('GT Color'), (10,20), 0, 0.5, (0,255,0))
            cv2.putText(frame, ('DT Color'), (90,20), 0, 0.5, (0,0,255))
            cv2.putText(frame, ('Detections: ' + str(fname)), (10,40), 0, 0.5, (255,255,255))

            # try and resize
            height, width = frame.shape[:2]
            if resize_height is not 0 and resize_width is not 0 :
                oframe = frame
            else :
                oframe = cv2.resize(frame,(width*4/5, height*4/5), interpolation = cv2.INTER_AREA)

            cv2.imshow('flirViewer',oframe)
            cv2.moveWindow('flirViewer', 100, 100)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


    def viewFxImage(self, isFneg=True, numOfImages=1):
        '''
            Based on FN or not, setup GT or DT image with bounding boxes and
            call the image viewer
        '''

        # setup test category Id  & scoreThreshold
        testCatId = self.testCatId
        scoreThr = self.scoreThr

        resize_width = self.resize_width
        resize_height = self.resize_height

        if(isFneg is True):
            fxImageIds = self.fnImageIds
        else :
            fxImageIds = self.fpImageIds

        #fxImageIds.sort()

        # setup the GT and DT contexts
        """
        Gt = self.Gt
        GtAnns = self.GtAnnotations
        GtImgs = self.GtImages
        Dt = self.Dt
        
        img2dtidx = {}
        img2gtidx = {}
        img2gtimidx = {}
        for kk in range(len(Dt)):
            if Dt[kk]['image_id'] not in img2dtidx.keys():
                img2dtidx[Dt[kk]['image_id']] = []
            img2dtidx[Dt[kk]['image_id']].append(kk)
        for kk in range(len(GtAnns)):
            if GtAnns[kk]['image_id'] not in img2gtidx.keys():
                img2gtidx[GtAnns[kk]['image_id']] = []
            img2gtidx[GtAnns[kk]['image_id']].append(kk)
        for kk in range(len(GtImgs)):
            if GtImgs[kk]['id'] not in img2gtimidx.keys():
                img2gtimidx[GtImgs[kk]['id']] = []
            img2gtimidx[GtImgs[kk]['id']].append(kk)
        """
        
        (Gt, GtAnnotations, GtImages, gtImageIds, Dt, dtImageIds, rangemax, img2dtidx, img2gtidx, img2gtimidx) = self.loadAssets()
        if numOfImages > -1: rangemax = min(numOfImages, rangemax)
        
        # Get the Detection and GT indices with the same imageID
        for jj in range(0, numOfImages):
            if (jj < 0):        # for debug only
                imgId = 32      # 1763 1772 1745 1746 1749
            else :
                imgId = fxImageIds[jj]
            # index_dt = self._find(Dt, 'image_id', imgId)
            # index_gt = self._find(GtAnns, 'image_id', imgId)
            try: index_dt = img2dtidx[imgId] # self._find(Dt, 'image_id', imgId)  
            except KeyError: index_dt = []
            try: index_gt = img2gtidx[imgId] # self._find(GtAnns, 'image_id', imgId)
            except KeyError: index_gt = []

            # get the image file name from GT file
            # index_fname = self._find(GtImgs, 'id', imgId)
            try: index_fname = img2gtimidx[imgId]
            except KeyError: index_fname = []
            #print 'index_fname : {}'.format(index_fname) + '\n'
            dic = GtImgs[int(index_fname[0])]
            fname = dic['file_name']
            #print 'image file name is: ' + fname + '\n'
            #print 'dic[] is {}\n'.format(dic)

            if resize_height is not 0 and resize_width is not 0 :
                image = cv2.imread(fname)
                frame = cv2.resize(image, (resize_height, resize_width))
            else :
                frame = cv2.imread(fname)

            # get the detection bbox information
            box_Dt = np.zeros((len(index_dt), 4))
            c=0
            for ind in index_dt:
                dic = Dt[ind]
                catId = dic['category_id']
                score = dic['score']

                # setup the txt to be displayed on the image
                displayStr = str(catId) + ", " +str(score)

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId) ) :
                    continue

                # only show the images if the score is >= the scoreThr
                if( (score < scoreThr) ) :
                    continue

                box_Dt[c,:] = dic['bbox']
                x = int(box_Dt[c,0])
                y = int(box_Dt[c,1])
                w = int(box_Dt[c,2])
                h = int(box_Dt[c,3])

                if (isFneg is False) :
                    # this is FP. So, no GT, but have DTs
                    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)
                else :
                    # else FN, does not have any DTs i.e., no-op
                    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)

                # TODO: post name instead of integer
                cv2.putText(frame, str(displayStr), (x, y-5), 0, 0.5, (255,255,0))
                c = c + 1

            # get the GT bbox information
            box_Gt = np.zeros((len(index_gt), 4))
            c=0
            for ind in index_gt:
                dic = GtAnnotations[ind]
                catId = dic['category_id']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId) ) :
                    continue

                box_Gt[c,:] = dic['bbox']
                x = int(box_Gt[c,0])
                y = int(box_Gt[c,1])
                w = int(box_Gt[c,2])
                h = int(box_Gt[c,3])

                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),1)
                # TODO: post name instead of integer
                cv2.putText(frame, str(catId), (x, y-5), 0, 0.5, (0,255,0))
                c = c + 1

            # Show the image along with the filename
            cv2.putText(frame, ('GT Color'), (10,20), 0, 0.5, (0,255,0))
            #cv2.putText(frame, ('DT Color'), (90,20), 0, 0.5, (255,255,0))
            cv2.putText(frame, ('DT Color'), (90,20), 0, 0.5, (0,0,255))
            if (isFneg is True) :
                cv2.putText(frame, ('FN: ' + str(fname)), (10,40), 0, 0.5, (255,255,255))
            else :
                cv2.putText(frame, ('FP: ' + str(fname)), (10,40), 0, 0.5, (255,255,255))

            # try and resize
            height, width = frame.shape[:2]
            #oframe = cv2.resize(frame,(width*2/3, height*2/3), interpolation = cv2.INTER_CUBIC)
            oframe = cv2.resize(frame,(width*4/5, height*4/5), interpolation = cv2.INTER_AREA)
            cv2.imshow('bboxes',oframe)
            cv2.moveWindow('bboxes', 100, 100)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


    def viewFalseNeg(self, inFname=None, numOfImages=-1):
        '''
            setup FN contexts, data and call the fxviewer method
        '''

        if (inFname is None):
            print 'Need False Negative Results file {}\n'.format(inFname)
            raise Exception('File NOT provided')
            #return 'File NOT provided'

        self.fnegFile = inFname
        print 'FN results Input File {}\n'.format(self.fnegFile)

        # get the false negative image IDs
        infile = open(inFname).read()
        data = json.loads(infile)
        fxdata = data['fndata']
        self.fndata = fxdata

        fxImageIds = []
        for i in range(len(fxdata)):
            fxImageIds.append(fxdata[i]['imageId'])

        fxImageIds = list(set(fxImageIds))
        fxImageIds.sort()
        self.fnImageIds = fxImageIds

        # setup number of images to view
        if (numOfImages > -1) :
            rangemax = numOfImages
        else :
            rangemax = len(fxImageIds)

        if (len(fxImageIds) != 0) :
            print 'FN NumOfImages {} and Viewing {}'.format(len(fxImageIds), rangemax)

            # check if there are enough images to view
            if (len(fxImageIds) < rangemax) :
                rangemax = len(fxImageIds)

            # Call the viewer and let us see what we got
            self.viewFxImage(True, rangemax)

        else:
            print 'There are ZERO FN!!'



    def viewFalsePos(self, inFname=None, numOfImages=-1):
        '''
            setup FP contexts, data and call the fxviewer method
        '''

        if (inFname is None):
            print 'Need False Positive Results file {}\n'.format(inFname)
            raise Exception('File NOT provided')
            #return 'File NOT provided'

        self.fposFile = inFname
        print 'FP results Input File {}\n'.format(self.fposFile)

        # get the false positive image IDs
        infile = open(inFname).read()
        data = json.loads(infile)
        fxdata = data['fpdata']
        self.fpdata = fxdata

        fxImageIds = []
        for i in range(len(fxdata)):
            fxImageIds.append(fxdata[i]['imageId'])

        fxImageIds = list(set(fxImageIds))
        fxImageIds.sort()
        self.fpImageIds = fxImageIds

        # setup number of images to view
        if (numOfImages > -1) :
            rangemax = numOfImages
        else :
            rangemax = len(fxImageIds)

        if (len(fxImageIds) != 0) :
            print 'FP NumOfImages {} and Viewing {}'.format(len(fxImageIds), rangemax)

            # check if there are enough images to view
            if (len(fxImageIds) < rangemax) :
                rangemax = len(fxImageIds)

            # Call the viewer and let us see what we got
            self.viewFxImage(False, rangemax)

        else:
            print 'There are ZERO FP!!'

    def findDtBboxForGtBbox(self, gb, gcatId, index_dt):
        '''
            check the DtBbox corresponding to the passed GtBbox
            and return the area of the GtBbox.
            NOTE: This is no longer called
        '''
        garea = gb[2]*gb[3]
        dtAnns = self.Dt_copy

        for ind in index_dt:

            if dtAnns[ind] is None : 
                continue

            db = dtAnns[ind]['bbox']
            darea = db[2]*db[3]

            catId = dtAnns[ind]['category_id']
            if catId != gcatId :
                continue

            # check if IoU is < 0.5 : if yes, ignore
            w = min(db[0]+db[2],gb[0]+gb[2]) - max(db[0],gb[0])
            h = min(db[1]+db[3],gb[1]+gb[3]) - max(db[1],gb[1])
            if w <= 0 or h <= 0 :
                continue

            iarea = float(w*h)
            denom = float(garea + darea - iarea)
            if denom <= 0.0 :
                continue

            iou = iarea/denom
            if iou < 0.5 :
                continue
            else : 
                #return iarea 
                dtAnns = [x for x in dtAnns if x != ind]
                self.Dt_copy = dtAnns
                return 1 

            # find if the rectangles intersect
            #   if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
            #       bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin())
            #           there is no intersection
            #if ((gb[0] > (db[0]+db[2])) or (db[0] > (gb[0]+gb[2])) or (gb[1] > (db[1]+db[3])) or (db[1] > (gb[1]+gb[3]))) :
            #    continue

        return 0


    def findGtBboxForDtBbox(self, db, dcatId, index_gt):
        '''
            check the GtBbox corresponding to the passed DtBbox
            and return the area of the GtBbox.
        '''

        #gtUsed = self.GtUsed

        GtAnns = self.GtAnnotations_copy
        darea = db[2]*db[3]

        for ind in index_gt:

            catId = GtAnns[ind]['category_id'] 
            if (catId != dcatId) :      # if detection catId is not the same as gt catId, ignore
                continue

            gb = GtAnns[ind]['bbox']
            garea = gb[2]*gb[3]

            # check if IoU is < 0.5 : if yes, ignore
            w = min(db[0]+db[2],gb[0]+gb[2]) - max(db[0],gb[0])
            h = min(db[1]+db[3],gb[1]+gb[3]) - max(db[1],gb[1])
            if w <= 0 or h <= 0 :
                continue
            iarea = float(w*h)
            denom = float(garea + darea - iarea)
            if denom <= 0.0 :
                continue

            iou = iarea/denom
            if iou < 0.5 :
                continue
            else:
                # keep track of matched GT
                GtAnns[ind]['category_id'] = 100000 + catId
                self.GtAnnotations_copy = GtAnns
                return garea,ind

        return 0,0


    def saveImagesWithDetections(self):
        '''
            Setup GT Annotations & DT bounding boxes on the image and
            save the image as a separate file
        '''

        saveImages = True
        if self.detectionImagesOutputDir is None :
            print '====================== Not saving images with bbox detections ====================='
            saveImages = False
            #return
        else :
            print '== Saving images with bbox detections in {}'.format(self.detectionImagesOutputDir)

        if (self.resJson is None or self.annJson is None):
            print 'DetResultsFile: {}'.format(self.resJson)
            print 'GT-AnnotationsFile: {}'.format(self.annJson)
            print 'Need to Initialize with above files'
            raise Exception('File NOT provided')
            #return 'File NOT provided'

        resize_width = self.resize_width
        resize_height = self.resize_height

        # setup test category Id  & scoreThreshold
        testCatId = self.testCatId
        scoreThr = self.scoreThr
        
        (Gt, GtAnnotations, GtImages, gtImageIds, Dt, dtImageIds, rangemax, img2dtidx, img2gtidx, img2gtimidx) = self.loadAssets()
        """
        # setup the contexts and load data
        resJson = self.resJson
        annJson = self.annJson

        # get the image IDs from the GT annotations
        ann = open(annJson).read()
        Gt = json.loads(ann)
        GtAnns = Gt['annotations']
        GtImgs = Gt['images']
        self.Gt = Gt
        self.GtAnnotations = GtAnns
        self.GtImages = GtImgs
        """
        
        self.GtAnnotations_copy = copy.deepcopy(GtAnnotations)
        self.Dt_copy = copy.deepcopy(Dt)

        ###
        ## save GT/TP/FP bbox co-ordinates and FN metadata in a separate json file.  
        ## Base this on the GT file and augment each annotation entry with: TP-bbox or FN meta-data
        ## and FPs on each image. Also: include an entry for per image count of TPs, FPs & FNs.
        ###
        statsDts = []   # has FP or no key 
        statsGts = []   # has FN or no key
        statsDataOut = []
        statsImages = copy.deepcopy(GtImages)
        
        """
        gtImageIds = []
        for i in range(len(GtImgs)):
            gtImageIds.append(GtImgs[i]['id'])

        gtImageIds = list(set(gtImageIds))
        gtImageIds.sort()
        self.gtImageIds = gtImageIds
        
        
        # get the image IDs from the detection results
        res = open(resJson).read()
        Dt = json.loads(res)
        self.Dt = Dt
        self.Dt_copy = copy.deepcopy(Dt)

        imgIds = []
        for i in range(len(Dt)):
	    imgIds.append(Dt[i]['image_id'])

        imgIds = list(set(imgIds))
        imgIds.sort()
        self.dtImageIds = imgIds

        ## let us cycle through all GT images & save the GT annotations and Dt results
        rangemax = len(gtImageIds)
        imgIds = gtImageIds
        """
        ## count the number of detections and annotations of each s/m/l
        gs = gm = gl = ds = dm = dl = 0
        sfp = mfp = lfp = sfn = mfn = lfn = 0

        ## keep a count of per image and overall FPs, TPs, FNs, GTs and Dts
        num_tps = num_fps = num_fns = total_tps = total_fps = total_fns = 0
        num_gts = total_gts = num_dts = total_dts = 0

        ## setup output dir for saving these images
        if saveImages is False : 
            detPathName, junkFileName = os.path.split(self.resJson)
            self.detectionImagesOutputDir = detPathName
        else : 
            detPathName = self.detectionImagesOutputDir
        print 'detPathName is {} '.format(detPathName)

        if not os.path.exists(detPathName):
            os.makedirs(detPathName)

        rfname, rfext = os.path.splitext(self.resJson)
        csvFileName = rfname + '_stats.csv'
        statsJsonFileName = rfname + '_tp_fp_fn_boxes.json'
        print 'GT/FP/TP/FN json stats file name is: ' + statsJsonFileName
        print 'csvFileName is: ' + csvFileName

        ## write into a csv file : image filename, number of TPs, number of FPs, number of FNs
        csvFile = open(csvFileName, 'w')
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        csvWriter.writerow(['image_file_name', 'image_id', 'GTs', 'TPs', 'FPs', 'FNs'])
        imgIds = gtImageIds
        # Get the Detection and GT indices with the same imageID
        for jj in range(rangemax):
            imgId = imgIds[jj]

            # index_dt = self._find(Dt, 'image_id', imgId)
            # index_gt = self._find(GtAnns, 'image_id', imgId)
            try: index_dt = img2dtidx[imgId] # self._find(Dt, 'image_id', imgId)  
            except KeyError: index_dt = []
            try: index_gt = img2gtidx[imgId] # self._find(GtAnns, 'image_id', imgId)
            except KeyError: index_gt = []

            # setup stats image context also
            #statsImageRecord = [c_row for c_row in cmp_data if c_row['image_id'] == image_id][0]
            statsImageRecord = [simg_row for simg_row in statsImages if simg_row['id'] == imgId][0]
            statsImageRecord['Dts'] = []
            statsImageRecord['Gts'] = []

            ## initialzie image specific counters
            num_tps = num_fps = num_fns = num_dts = num_gts = 0

            # (km) get the image file name from GT file 
            # index_fname = self._find(GtImgs, 'id', imgId)
            try: index_fname = img2gtimidx[imgId]
            except KeyError: index_fname = []
            dic = GtImages[int(index_fname[0])]
            fname = dic['file_name']

            # name for imagefile to be saved with detection & gt boxes
            pathName, fileName = os.path.split(fname)

            ## ==================================================
            ##      resize for map-prime case
            ## ==================================================
            if resize_height is not 0 and resize_width is not 0 :
	        image = cv2.imread(fname)
                frame = cv2.resize(image, (resize_height, resize_width))

                if len(index_dt) is 0 :
                    pstr = '/npdet-'
                else :
                    pstr = '/pdet-'

                toSaveImgFileName = detPathName + pstr + fileName

            else :
	        frame = cv2.imread(fname)

                if len(index_dt) is 0 :
                    pstr = '/ndet-'
                else :
                    pstr = '/det-'

                pstr = '/det-'  ## HACK XXX : To get the file names in order for video

                toSaveImgFileName = detPathName + pstr + fileName

            #print 'image file name is: '+fname+'toSaveImgFileName: '+toSaveImgFileName+'\n'

            # get the detection bbox information
	    box_Dt = np.zeros((len(index_dt), 4))
	    c=0
	    for ind in index_dt:
                dic = Dt[ind]
                catId = dic['category_id']
                score = dic['score']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    continue

                isFP = False
                box_Dt[c,:] = dic['bbox']
                x = int(box_Dt[c,0])
                y = int(box_Dt[c,1])
                w = int(box_Dt[c,2])
                h = int(box_Dt[c,3])
                area = int(w*h)

                ## find the corresponding gt bbox and count based on the gt bbox size
                garea,matchedIndex = self.findGtBboxForDtBbox(dic['bbox'], catId, index_gt)
                if garea == 0 :
                    astring = 'all'
                    isFP = True

                # this should be based on gt area
                if garea > 0 and garea <= 32*32 :
                    astring = 'small'
                    ds += 1
                    if isFP is True :
                        sfp += 1
                elif garea > 96*96 :
                    astring = 'large'
                    dl += 1
                    if isFP is True :
                        lfp += 1
                elif garea > 32*32 and garea <= 96*96:
                    astring = 'medium'
                    dm += 1
                    if isFP is True :
                        mfp += 1

                if isFP is True : 
                    ## NOTE : The number of FPs could be in 100s per image when confThr is at 0.01 (default). 
                    ##          So, put the FP box only when confThr is > 0.2
                    if scoreThr > 0.2 : 
                        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,0),1)

                    displayStr = str(catId)
                    #displayStr = '*'
                    cv2.putText(frame, str(displayStr), (x, y-5), 0, 0.1, (225,225,0))
                    num_fps += 1
                else :
                    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)
                    displayStr = str(catId)
                    cv2.putText(frame, str(displayStr), (x, y-5), 0, 0.3, (0,255,255))
                    num_tps += 1

                dic['isFP'] = isFP
                dic['size'] = astring
                statsImageRecord['Dts'].append(dic)

                num_dts += 1
                c = c + 1

            # get the GT bbox information
            GtAnns_copy = self.GtAnnotations_copy
            box_Gt = np.zeros((len(index_gt), 4))
            c=0
            for ind in index_gt:
                dic = GtAnnotations[ind]
                catId = dic['category_id']

                dic_copy = GtAnns_copy[ind]
                catId_copy = dic_copy['category_id']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    continue

                box_Gt[c,:] = dic['bbox']
                x = int(box_Gt[c,0])
                y = int(box_Gt[c,1])
                w = int(box_Gt[c,2])
                h = int(box_Gt[c,3])
                area = int(w*h)

                ## find if this GT was matched with a detection or not & count accordingly
                if (catId == (catId_copy - 100000)) : 
                    isFN = False
                else : 
                    # this was not matched to a detection
                    isFN = True

                if area <= 32*32 :
                    astring = 'small'
                    gs += 1
                    if isFN is True :
                        sfn += 1
                elif area > 96*96 :
                    astring = 'large'
                    gl += 1
                    if isFN is True :
                        lfn += 1
                else :
                    astring = 'medium'
                    gm += 1
                    if isFN is True :
                        mfn += 1

                # setup the txt to be displayed on the image
                displayStr = str(catId)

                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),1)
                cv2.putText(frame, str(displayStr), (x, y+h+10), 0, 0.3, (255,0,255))

                # update stats record with this GT/FN info
                dic['isFN'] = isFN
                dic['size'] = astring
                statsImageRecord['Gts'].append(dic)

                num_gts += 1
                c = c + 1

            # (km) : Show the image filename also.
            #	 Input args for putText:
            #	    image, text, coordinates_for_start_of_text, font_type, font_scale, font_color
            num_fns = num_gts - num_tps
            if saveImages is True : 
                countString = 'Number of GTs: {}, TPs {}, FNs {}, FPs {}'.format(num_gts, num_tps, num_fns, num_fps)
                cv2.putText(frame, ('GT Color'), (10,20), 0, 0.4, (0,255,0))
                cv2.putText(frame, ('DT Color'), (90,20), 0, 0.4, (0,0,255))
                cv2.putText(frame, ('FP Color'), (170,20), 0, 0.4, (255,255,0))
                cv2.putText(frame, (str(fileName)), (10,40), 0, 0.4, (255,255,255))
                cv2.putText(frame, (str(countString)), (10,60), 0, 0.4, (255,255,255))

                oframe = frame

                # save the image with detection & gt boxes into a file
                cv2.imwrite(toSaveImgFileName, oframe)

            ## book keeping : update overall counters from image specific ones
            total_tps += num_tps
            total_fps += num_fps
            total_fns += num_fns
            total_gts += num_gts
            total_dts += num_dts

            ## update the stats image record also and append into dataout
            statsImageRecord['numGTs'] = num_gts
            statsImageRecord['numDTs'] = num_dts
            statsImageRecord['numTPs'] = num_tps
            statsImageRecord['numFPs'] = num_fps
            statsImageRecord['numFNs'] = num_fns
            statsDataOut.append(statsImageRecord)

            ## write into a csv file : image filename full path, image_id, number of GTs, TPs, FPs, FNs
            csvWriter.writerow([toSaveImgFileName, imgId, num_gts, num_tps, num_fps, num_fns])

        # write statsDataOut into the stats json file
        with open(statsJsonFileName, 'w') as outfile:
            json.dump(statsDataOut, outfile, indent=4, sort_keys=True, separators=(',', ':'))

        # close csv file
        csvFile.close()

        # print the s/m/l detection sizes also from each
        dstr =  'Counts Dts S/M/L : {}, {}, {} '.format(ds, dm, dl)
        gstr =  'Counts Gts S/M/L : {}, {}, {} '.format(gs, gm, gl)
        fpstr = 'Counts FPs S/M/L : {}, {}, {} '.format(sfp, mfp, lfp)
        fnstr = 'Counts FNs S/M/L : {}, {}, {} '.format(sfn, mfn, lfn)
        cnstr = 'Number of GTs: {}, TPs {}, FNs {}, FPs {}'.format(total_gts, total_tps, total_fns, total_fps)

        if saveImages is True : 
            cv2.putText(oframe, str(dstr), (10,80), 0, 0.4, (255,255,255))
            cv2.putText(oframe, str(gstr), (10,100), 0, 0.4, (255,255,255))
            cv2.putText(oframe, str(fpstr), (10,120), 0, 0.4, (255,255,255))
            cv2.putText(oframe, str(fnstr), (10,140), 0, 0.4, (255,255,255))
            cv2.putText(oframe, str(cnstr), (10,160), 0, 0.4, (255,255,255))

            toSaveImgFileName = toSaveImgFileName + 'counts_'
            toSaveImgFileName = detPathName + '/counts_' + fileName
            cv2.imwrite(toSaveImgFileName, oframe)


# ==================================================================================
#
#
# ==================================================================================
    def dumpFPandFNcountsByArea(self):
        '''
            Count the FPs and FNs by size of the bbox of GT or Dt respectively
                and log them out. Also: save them in a json file
        '''

        print '===================== FP and FN counts based on bbox sizes ========================'

        if (self.resJson is None or self.annJson is None):
            print 'DetResultsFile: {}'.format(self.resJson)
            print 'GT-AnnotationsFile: {}'.format(self.annJson)
            print 'Need to Initialize with above files'
            raise Exception('File NOT provided')

        # setup test category Id  & scoreThreshold
        testCatId = self.testCatId
        confThr = self.scoreThr
        
        # setup the contexts and load data
        resJson = self.resJson
        annJson = self.annJson
        """
        # get the image IDs from the GT annotations
        ann = open(annJson).read()
        Gt = json.loads(ann)
        GtAnns = Gt['annotations']
        GtImgs = Gt['images']
        self.Gt = Gt
        self.GtAnnotations = GtAnns
        self.GtAnnotations_copy = copy.deepcopy(Gt['annotations'])
        self.GtImages = GtImgs

        gtImageIds = []
        for i in range(len(GtImgs)):
            gtImageIds.append(GtImgs[i]['id'])

        gtImageIds = list(set(gtImageIds))
        gtImageIds.sort()
        self.gtImageIds = gtImageIds

        # get the image IDs from the detection results
        res = open(resJson).read()
        Dt = json.loads(res)
        self.Dt = Dt
        self.Dt_copy = copy.deepcopy(Dt)

        imgIds = []
        for i in range(len(Dt)):
	    imgIds.append(Dt[i]['image_id'])

        imgIds = list(set(imgIds))
        imgIds.sort()
        self.dtImageIds = imgIds

        ## let us cycle through all GT images & save the GT annotations and Dt results
        rangemax = len(gtImageIds)
        imgIds = gtImageIds
        """
        
        (Gt, GtAnnotations, GtImages, gtImageIds, Dt, dtImageIds, rangemax, img2dtidx, img2gtidx, img2gtimidx) = self.loadAssets()
        self.GtAnnotations_copy = copy.deepcopy(GtAnnotations)
        self.Dt_copy = copy.deepcopy(Dt)
        imgIds = gtImageIds
        ## count the number of detections and annotations of each s/m/l
        gs,gm,gl,ds,dm,dl,stp,mtp,ltp = 0,0,0,0,0,0,0,0,0
        sfp,mfp,lfp,sfn,mfn,lfn = 0,0,0,0,0,0
        numGts,numDts,numFps,numFns,numTps = 0,0,0,0,0
        igt,idt,ifp,ifn,itp = 0,0,0,0,0

        # Get the Detection and GT indices with the same imageID
        # save the image with GT but no Detections as well i.e., FNs
        imgIds = gtImageIds
        for jj in range(rangemax):
            imgId = imgIds[jj]

            # index_dt = self._find(Dt, 'image_id', imgId)
            # index_gt = self._find(GtAnns, 'image_id', imgId)
            try: index_dt = img2dtidx[imgId] # self._find(Dt, 'image_id', imgId)  
            except KeyError: index_dt = []
            try: index_gt = img2gtidx[imgId] # self._find(GtAnns, 'image_id', imgId)
            except KeyError: index_gt = []

            # get the image file name from GT file
            try: index_fname = img2gtimidx[imgId]
            except KeyError: index_fname = []
            dic = GtImages[int(index_fname[0])]
            imageFileName = dic['file_name']

            # initialize image specific counters
            itp,ifp,ifn,idt,igt = 0,0,0,0,0

            # get the detection bbox information
	    box_Dt = np.zeros((len(index_dt), 4))
	    c=0
	    for ind in index_dt:
                dic = Dt[ind]
                catId = dic['category_id']
                score = dic['score']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    continue

                isFP = False

                box_Dt[c,:] = dic['bbox']
                x = int(box_Dt[c,0])
                y = int(box_Dt[c,1])
                w = int(box_Dt[c,2])
                h = int(box_Dt[c,3])
                area = w*h

                ## find the corresponding gt bbox and count based on the gt bbox size
                garea,matchedIndex = self.findGtBboxForDtBbox(dic['bbox'], catId, index_gt)

                # if there is no corresponding GT, need to figure out the area
                if garea == 0 :
                    isFP = True
                    ifp += 1
                else :
                    itp += 1
                    area = garea        ## account TP against GT's bbox size

                if area < 32*32 :
                    ds += 1
                    if isFP is True :
                        sfp += 1
                    else:
                        stp += 1
                elif area > 96*96 :
                    dl += 1
                    if isFP is True :
                        lfp += 1
                    else:
                        ltp += 1
                else :
                    dm += 1
                    if isFP is True :
                        mfp += 1
                    else:
                        mtp += 1

                idt += 1
                c = c + 1

            # get the GT bbox information
            GtAnns_copy = self.GtAnnotations_copy
            box_Gt = np.zeros((len(index_gt), 4))
            c=0
            for ind in index_gt:
                dic = GtAnnotations[ind]
                catId = dic['category_id']

                dic_copy = GtAnns_copy[ind]
                catId_copy = dic_copy['category_id']

                # only do this for the testCatId if provided
                if( (testCatId > 0) and (catId != testCatId)) :
                    #print 'No Match: catId is {} and testCatId is {}'.format(catId, testCatId)
                    continue

                box_Gt[c,:] = dic['bbox']
                x = int(box_Gt[c,0])
                y = int(box_Gt[c,1])
                w = int(box_Gt[c,2])
                h = int(box_Gt[c,3])
                area = int(w*h)

                ## find if this GT was matched with a detection or not & count accordingly
                if (catId == (catId_copy - 100000)) : 
                    isFN = False
                else : 
                    # this was not matched to a detection
                    isFN = True

                #print 'catId {}, copy {}, testCatId {}, isFN {}'.format(catId, catId_copy, testCatId, isFN)

                if area < 32*32 :
                    gs += 1
                    if isFN is True :
                        sfn += 1
                elif area > 96*96 :
                    gl += 1
                    if isFN is True :
                        lfn += 1
                else :
                    gm += 1
                    if isFN is True :
                        mfn += 1

                igt += 1
                c = c + 1

            # book keeping: update overall counters from image specific counts
            ifp = idt - itp
            ifn = igt - itp
            numTps += itp
            numDts += idt
            numGts += igt
            numFps += ifp
            numFns += ifn

        # print the s/m/l detection sizes also from each
        print 'Confidence Threshold        : %0.2f'%(confThr)
        print 'Category ID                 : %7d '%(testCatId)
        print 'Total Number of GT          : %7d '%(numGts)
        print 'Total Number of Detections  : %7d '%(numDts)
        print 'Total Number of (TP+FP)     : %7d '%((numTps + numFps))
        print 'Total Number of TP          : %7d '%(numTps)
        print 'Total Number of FP          : %7d '%(numFps)
        print 'Total Number of FN          : %7d '%(numFns)
        print 'Counts GT S/M/L             : %7d, %7d, %7d  Total: %7d'%(gs, gm, gl, (gs+gm+gl))
        print 'Counts Detections S/M/L     : %7d, %7d, %7d  Total: %7d'%(ds, dm, dl, (ds+dm+dl))
        print 'Counts TP S/M/L             : %7d, %7d, %7d  Total: %7d'%(stp, mtp, ltp, (stp+mtp+ltp))
        print 'Counts FP S/M/L             : %7d, %7d, %7d  Total: %7d'%(sfp,mfp,lfp, (sfp+mfp+lfp))
        print 'Counts FN S/M/L             : %7d, %7d, %7d  Total: %7d'%(sfn,mfn,lfn, (sfn+mfn+lfn))
        print "==================================================================================="

    def loadAssets(self):
        if self.resJson == None or self.annJson == None:
            return
        if self.annLoaded == [ self.resJson , self.annJson ]:
            return self.dumpData()
        
        resJson = self.resJson
        annJson = self.annJson
        print 'Loading {} {} ...'.format(resJson, annJson)
        # get the image IDs from the GT annotations
        ann = open(annJson).read()
        Gt = json.loads(ann)
        GtAnns = Gt['annotations']
        GtImgs = Gt['images']
        self.Gt = Gt
        self.GtAnnotations = GtAnns
        self.GtAnnotations_copy = copy.deepcopy(Gt['annotations'])
        self.GtImages = GtImgs
        
        gtImageIds = []
        for i in range(len(GtImgs)):
            gtImageIds.append(GtImgs[i]['id'])
        
        gtImageIds = list(set(gtImageIds))
        gtImageIds.sort()
        self.gtImageIds = gtImageIds
        
        # get the image IDs from the detection results
        res = open(resJson).read()
        Dt = json.loads(res)
        self.Dt = Dt
        
        imgIds = []
        for i in range(len(Dt)):
            imgIds.append(Dt[i]['image_id'])
        
        imgIds = list(set(imgIds))
        imgIds.sort()
        self.dtImageIds = imgIds
        
        # setup number of images to view
        """
        if (numOfImages > -1) :
                rangemax = numOfImages
        else :
                rangemax = len(imgIds)
        """
        rangemax = len(imgIds)
        # check if there are enough images to view
        if (len(imgIds) < rangemax) :
                rangemax = len(imgIds)
        
        self.rangemax = rangemax
        
        img2dtidx = {}
        img2gtidx = {}
        img2gtimidx = {}
        for kk in range(len(Dt)):
                if Dt[kk]['image_id'] not in img2dtidx.keys():
                        img2dtidx[Dt[kk]['image_id']] = []
                img2dtidx[Dt[kk]['image_id']].append(kk)
        for kk in range(len(GtAnns)):
                if GtAnns[kk]['image_id'] not in img2gtidx.keys():
                        img2gtidx[GtAnns[kk]['image_id']] = []
                img2gtidx[GtAnns[kk]['image_id']].append(kk)
        for kk in range(len(GtImgs)):
                if GtImgs[kk]['id'] not in img2gtimidx.keys():
                        img2gtimidx[GtImgs[kk]['id']] = []
                img2gtimidx[GtImgs[kk]['id']].append(kk)
        
        self.img2dtidx = img2dtidx
        self.img2gtidx = img2gtidx
        self.img2gtimidx = img2gtimidx
        self.annLoaded = [ self.resJson , self.annJson ]
        
        return self.dumpData()
    
    def dumpData(self):
        return (self.Gt, self.GtAnnotations, self.GtImages, self.gtImageIds, self.Dt, self.dtImageIds, self.rangemax, self.img2dtidx, self.img2gtidx, self.img2gtimidx)
