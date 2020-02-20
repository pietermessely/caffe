#from __future__ import print_function

"""
Q2'2018:   Created
@author:   kedar m

"""

'''
    This will merge a foreground image (using segmented mask) onto another background image.
        It will resize the bg image to fg image size when requried

    Mandatory* and Optional# Arguments & Outputs:
    -------------------------------
        a*. Absolute path of source directory with fg images (e.g., /mnt/data/conservator/fgImages)
            - This Program will assume sub-dirs "Annotations" & "Data" are present in the above dir
        b*. Absolute path of 'background" images directory (e.g., /mnt/data/conservator/bgImages)
            - This program will assume this directory has images that can be used for background
            - The program will pick an image at random from this directory
        c*. Absolute path of the 'output' directory where the final images & annotations will be posted
            - e.g., /mnt/data/conservator/overlay
            - The program will create "Annotations" & "Data" under the above dir to post the
                relevant annotations and images. A readme txt file will contain the date, fg & bg 
                directory paths that were used resulting in this dataset.
            - The output directory can then be used as src to create LMDB files for training or
                as 'test' dataset. 
        d#. colorBW : color for foreground images is black/white(1) or multi-color (0)
        e#. view : view final overlay images (1) or save them in the dst dir (0)
        f#. Final overlay Filename prefix : prefix to use for naming the final overlay files
'''
import os
import json
import shutil
import string
import sys
import argparse
import random
import glob
import numpy as np
import cv2

## ================================
# Defaults 
## =================================

#fgImage = 'fg-rgb.jpg'
#segImage = 'fg-seg.jpg'
#bgImage = 'bg.jpg'

random.seed(100)

preDefinedColorsBW2 = {
    "0" : [0,0,0],                      # black
    "1" : [255,255,255],                # white
    "2" : [245,245,245],                # white smoke
    "3" : [225,225,225],                # gray
}

preDefinedColorsBW = {
    "0" : [245,245,245],                # white smoke
    "1" : [225,225,225],                # gray
}

preDefinedColors = {
    "0" : [0,0,0],                      # black
    "1" : [255,255,255],                # white
    "2" : [0,255,0],                    # green
    "3" : [255,0,0],                    # blue
    "4" : [0,0,255],                    # red
    "5" : [255,0,255],                  # magenta
    "6" : [0,255,255],                  # yellow
    "7" : [255,255,0],                  # shade of blue
    "8" : [50,50,50],                   # 
    "9" : [200,200,200],                # 
    "10": [245,245,245],                # white smoke
    "11": [225,225,225],                # gray
}
numOfPreDefinedColors = len(preDefinedColors)
maxColors = numOfPreDefinedColors

#maxColors = 16
#maxColors = 4

debugView = 0           # default to disable i.e., will write the file
useBWcolors = 1         # default, yes. 

overlayFileNamePrefix = 'overlay'

## ================================================================
#       Input Arguments
## =================================================================

def getInputArgs(argv) :
    
    inOptions = {}
    while argv: 
        if argv[0][0] == '-':
            inOptions[argv[0]] = argv[1]
        argv = argv[1:]

    return inOptions


## =============================================
#       using segmentation masks of fgImage
## =============================================


def fgOnBgImageOverlay(fgImage, bgImage, dstImage, debugView, colorIdx):

#   Read the Image and get image sizes. Ensure same size for all.
    #fimg = cv2.imread(fgImage)#, cv2.IMREAD_UNCHANGED) 
    segimg = cv2.imread(fgImage)#, cv2.IMREAD_UNCHANGED) 
    bimg = cv2.imread(bgImage)#, cv2.IMREAD_UNCHANGED)
    #segimg = cv2.imread(segImage)#, cv2.IMREAD_UNCHANGED) 

    #fh,fw = fimg.shape[:2]
    sh,sw = segimg.shape[:2]
    bh,bw = bimg.shape[:2]

    if (sh != bh) or (sw != bw) : 
        print 'FG & BG Image aspect ratios are not the same. Resizing BG Image'
        bimg = cv2.resize(bimg, (sw,sh))
        #return

    segimg_gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(segimg_gray, 64, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)

    # After inv_mask, outline contours of the objects is highlighted
    # use this as required - in future
    #img1 = cv2.bitwise_and(segimg, segimg, mask = inv_mask)
    #cv2.imshow("seg-Img-after-mask", img1)
    #cv2.waitKey(0)

    # The bg image after applying the inv_mask, will show black pixels at the mask
    img2 = cv2.bitwise_and(bimg, bimg, mask = inv_mask)

    # try changing the color of the masked area from black to something else
    i = j = 0
    color = []
    #i = random.randint(0, numOfPreDefinedColors)

    if colorIdx%maxColors < numOfPreDefinedColors: 
        j = colorIdx%numOfPreDefinedColors
        color = preDefinedColors[str(j)]
    else :
        random.seed(colorIdx)
        color = [random.randint(100,200),random.randint(100,200),random.randint(100,200)],

    #color = preDefinedColors[str(colorIdx)]

    # copying the orig image for test purposes - so that we can see each
    img2_copy = img2.copy()
    img2_copy[np.where((img2_copy == [0,0,0]).all(axis = 2))] = color

    if debugView is 1: 
        imgidstr='bg-image id: ' + str(i)
        cv2.namedWindow(imgidstr)
        cv2.moveWindow(imgidstr, 100,100)
        cv2.imshow(imgidstr, img2_copy)
        cv2.waitKey(0)
    else :
        #outImageName='overlayImage_' + str(i)
        cv2.imwrite(dstImage, img2_copy)

    #displayStr = 'Finished Overlaying ' + str(i) + 'images'
    #print displayStr


if __name__ == '__main__' :
    
    from sys import argv
    inArgs = getInputArgs(argv)

    if '-f' in inArgs:
        fgImageDir = inArgs['-f']
    else :
        raise ValueError ('No Foreground Image directory provided')
 
    if '-b' in inArgs:
        bgImageDir = inArgs['-b']
    else :
        raise ValueError ('No Background Overlay Image(s) directory provided')
 
    if '-d' in inArgs:
        dstDir = inArgs['-d']
    else :
        raise ValueError ('No Destination directory provided')
 
    if '-p' in inArgs:
        overlayFileNamePrefix = inArgs['-p']
 
    if '-v' in inArgs: 
        debugView = int(inArgs['-v'])
    if not debugView is 1 : 
        debugView = 0

    if '-c' in inArgs: 
        useBWcolors = int(inArgs['-c'])

        if not useBWcolors is 1 : 
            useBWcolors = 0
            print 'Foreground Images will be of various colors'
            print 'numOfPreDefinedColors & Max Colors are: %d, %d '%(numOfPreDefinedColors, maxColors)
        else :
            print 'Foreground Images will only use black / white colors'
            numOfPreDefinedColors = len(preDefinedColorsBW)
            maxColors = numOfPreDefinedColors
            print 'numOfPreDefinedColors & Max Colors are: %d, %d '%(numOfPreDefinedColors, maxColors)

    if '-h' in inArgs: 
        print ('Usage: -f foreground Images dataset full path [e.g., /tmp/fgImage/]')
        print ('                It should contain Annotations & Data subdirs')
        print ('       -b background Images Dir full path [e.g., /tmp/bgImage/]')
        print ('       -d destination dataset Dir full path [e.g., /tmp/overlay/')
        print ('                Annotations & Data subdirs will be created')
        print ('       -h help [e.g., 0..100 - print help')
        print ('       -v view images [e.g., True or False]')
        print ('       -c use only black & white colors for foreground image [e.g., 1 or 0]')

    # weed out other errors
    if not os.path.isdir(fgImageDir): # or not os.path.isfile(fgImage):
        raise ValueError ('Invalid Foreground Segmentation Image Dir not Found')

    if not os.path.isdir(bgImageDir): # or not os.path.isfile(bgImage):
        raise ValueError ('Invalid Background Image Dir not Found')

    if not os.path.isdir(dstDir) :
        print ('creating destination Image directory')
        os.mkdir(dstDir)
    else :
        print ('Staging destination Image directory - clearing files in relevant sub-dirs')
        
        
    # clean up the destination dir first
    dstAnnoDir = dstDir + '/Annotations'
    dstDataDir = dstDir + '/Data'
    if os.path.exists(dstAnnoDir) :
        shutil.rmtree(dstAnnoDir)
    if os.path.exists(dstDataDir) :
        shutil.rmtree(dstDataDir)
    
    # create them fresh
    os.mkdir(dstAnnoDir)
    os.mkdir(dstDataDir)

    # setup the fgAnno and fgData paths & gather the files into a list
    fgAnnoDir = fgImageDir + '/Annotations'
    fgDataDir = fgImageDir + '/Data'
    fgAnnoList = glob.glob(fgAnnoDir + '/*')
    fgImageList = glob.glob(fgDataDir + '/*')

    numOfFgAnnos = len(fgAnnoList)
    numOfFgImages = len(fgImageList)
    if numOfFgAnnos is 0 or numOfFgImages is 0 :
        raise ValueError ('No Files exists in Foreground Images or Annotations dir')

    print('Number of Annotations and Images in fgDataset dir are: %d %d '%(numOfFgAnnos,numOfFgImages))
        
    if (numOfFgAnnos != numOfFgImages) :
        print('Number of Annotations and Images in fgDataset dir differ!!')
        print('Continuing as best effort')

    # get the fg anno extension
    #dummyFile = fgImageList[0]
    #dfn, srcAnnoExt = os.path.splitext(dummyFile)
    
    # setup the destination filenames full path minus the seqNum suffix & extension
    #dstImageExt = '.jpeg'
    #dstImagePrefix = dstDataDir + '/overlay_'
    #dstAnnoExt = srcAnnoExt
    #dstAnnoPrefix = dstAnnoDir + '/overlay_'
    
    # grab bg images in a list 
    bgImageList = glob.glob(bgImageDir + '/*')
    numOfBgImages = len(bgImageList)
    print 'number of bgImages: %d '%(numOfBgImages)

    print 'Final oerlay files will be tagged with the prefix: %s '%(overlayFileNamePrefix)

    # for each fg Image, pick a random bg image and save final image in dstDir with
    #   imageName as: overlay_{$seqNum}.jpeg
    for idx in range(0, numOfFgImages) : 
        fgImageFile = fgImageList[idx]
        fgAnnoFile = fgAnnoList[idx]

        j = idx % numOfBgImages
        bgImageFile = bgImageList[j]

        if not os.path.isfile(fgImageFile) : 
            continue

        if not os.path.isfile(fgAnnoFile) : 
            continue

        if not os.path.isfile(bgImageFile) : 
            continue
    
        fgifp, fgifn = os.path.split(fgImageFile)
        fgifile, fgiext = os.path.splitext(fgifn)
        fgafp, fgafn = os.path.split(fgAnnoFile)
        fgafile, fgaext = os.path.splitext(fgafn)
        dstImageFile = dstDataDir + '/' + overlayFileNamePrefix + '_' + fgifn
        dstAnnoFile = dstAnnoDir + '/' + overlayFileNamePrefix + '_' + fgafn
        #dstImageFile = dstDataDir + '/' + overlayFileNamePrefix + '_' + str(idx) + fgiext
        #dstAnnoFile = dstAnnoDir + '/' + overlayFileNamePrefix + '_' + str(idx) + fgaext

        # overlay the fg image mask onto the bgImage and save into dstImage
        fgOnBgImageOverlay(fgImageFile, bgImageFile, dstImageFile, debugView, colorIdx=idx)
        # copy the annotation file also
        shutil.copy(fgAnnoFile, dstAnnoFile)

        if (idx > 0) and ((idx % 1000) is 0) : 
            print 'Num Files Processed : %d'%(int(idx + 1))

    print 'Num Of Files Processed / Overlayed : %d'%(int(idx + 1))
    

# ==============================================
#       BELOW CODE NOT BEING USED FOR NOW
# ==============================================

## ==========================================
#       3-channels image matte using masks
#               doesn't work yet
## ========================================
if False : 

    ##orig_mask = load_seg_image()
    ##orig_mask_gray = covert_to_grayscale(orig_mask)
    #fimg_gray = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
    segimg_gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)

    ##
    ##mask = orig_mask_gray > 128
    ret,mask = cv2.threshold(segimg_gray, 64, 255, cv2.THRESH_BINARY)
    mask = np.array(mask > 0, dtype='uint8')
    #ret,inv_mask = cv2.threshold(fimg_gray, 127, 255, cv.THRESH_BINARY_INV)
    inv_mask = cv2.bitwise_not(mask)
    inv_mask = np.array(inv_mask > 0, dtype='uint8')

    ##fg = load_foreground_drone_image()
    ##bg = load_arbitrary_background_image()
    ##img1 = cv2.multiply(fg, mask)
    #img2 = cv2.multiply(bg, 1 - mask)
    #alpha_matted = img1 + img2
    #img1 = cv2.multiply(fimg, mask)
    #img2 = cv2.multiply(bimg, inv_mask)

    fg = fimg.copy();
    #cv2.imshow("fimgcopy", fg)
    fg[:,:,0] *= mask
    fg[:,:,1] *= mask
    fg[:,:,2] *= mask
    #cv2.imshow("fgmasked", fg)

    bg = bimg.copy()
    bg[:,:,0] *= inv_mask
    bg[:,:,1] *= inv_mask
    bg[:,:,2] *= inv_mask
    #cv2.imshow("fg", fg)
    #cv2.imshow("bg", bg)
    output = fg + bg

    if False:
        img1 = cv2.bitwise_and(fimg, fimg, mask = mask)
        img2 = cv2.bitwise_and(bimg, bimg, mask = inv_mask)
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        output = cv2.add(img1, img2)

    # show the image
    cv2.imshow("3-img-mask", output)
    cv2.waitKey(0)

## =====================
# works but commented out for now - better way is to get 
# the bbox area of the foreground image instead of the entire image?     
## =====================

if False: 
    #add extra dimension to the bimg
    baimg = np.dstack([bimg, np.ones((bh, bw), dtype='uint8') * 255])
    segaimg = np.dstack([segimg, np.ones((bh, bw), dtype='uint8') * 255])
    #add extra dimension to the fimg
    faimg = np.dstack([fimg, np.ones((fh, fw), dtype='uint8') * 255])

    # setup overlay and output image
    overlay = faimg.copy()
    output = baimg.copy()
    segcopy = segaimg.copy()

    # blend the two images i.e., (output = alpha*img1 + beta*img2 + gamma)
    # alpha value for transparency of foreground image 1(opaque), 0(transparent)
    alpha = 0.25      
    output1 = cv2.addWeighted(overlay, alpha, output, 1.0, 0)
    output2 = cv2.addWeighted(segcopy, alpha, output, 1.0, 0)
    #output = cv2.addWeighted(overlay, alpha, output, (1 - alpha), 0)

    # show the image
    cv2.imshow("alpha-channel", output1)
    cv2.waitKey(0)
    cv2.imshow("alpha-channel-seg", output2)
    cv2.waitKey(0)

## =====================
# works but commented out for now - better way so that we can just get 
# the bbox area of the foreground image instead of the entire image?     
## =====================

if False :      
  # loop over the alpha transparency values
  for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    # setup overlay and output image
    overlay = fimg.copy()
    output = bimg.copy()
 
    # TODO: grab cut the bbox rectangle from the overlay (foreground) image
    #cv2.rectangle(overlay, (420, 205), (595, 385), (0, 0, 255), -1)
    #cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha),
    #	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # show the output image
    print("alpha={}, beta={}".format(alpha, 1 - alpha))
    cv2.imshow("Output", output)
    cv2.waitKey(0)

## ======== This does not work ======= START ========
#
# mask = np.zeros(simg.shape[:2], np.uint8)
# grab the src image channels & get the alpha channel using the mask
#r_channel, g_channel, b_channel = cv2.split(simg)
#a_channel = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
#alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 # dummy alpha_image
#
# grab the dst image channels
#dr_channel, dg_channel, db_channel = cv2.split(dimg)
#
#img_RGBA = cv2.merge((dr_channel, dg_channel, db_channel, a_channel))
#img_BGRA = cv2.merge((db_channel, dg_channel, dr_channel, a_channel))
#img_RGBA = cv2.merge((dr_channel, dg_channel, db_channel, r_channel))
#img_BGRA = cv2.merge((db_channel, dg_channel, dr_channel, r_channel))
#
#cv2.namedWindow("atest")
#cv2.imshow("atest", img_RGBA)
#cv2.waitKey(0)
#
#cv2.namedWindow("btest")
#cv2.imshow("btest", img_BGRA)
#cv2.waitKey(0)
#
## ======== This does not work with our jpg/jpeg. Need 'png' type ==============
#
#(B, G, R, A) = cv2.split(fimg)  # this fails
#B = cv2.bitwise_and(B, B, mask=A)
#G = cv2.bitwise_and(G, G, mask=A)
#R = cv2.bitwise_and(R, R, mask=A)
#fimg = cv2.merge([B, G, R, A])
#
#cv2.imshow("fimg", fimg)
#cv2.waitKey(0)
#
#
## ======== This does not work ====== END ========

