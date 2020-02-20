import glob
import cv2
import numpy as np
import random
import argparse
import os
import shutil

'''
Written by ZTane Q3 2018
Cominbed Evan's scripts into a single script and set up arguments
Update: Major refactoring including more efficent chaining, additional augmentations, ability to write separate 
output without altering input, and random sampling. 
'''

np.random.seed(0)

def bilateralAugmentation(img_lst):
    total = str(len(img_lst))
    j = 0
    for i in img_lst:
        img = cv2.imread(i)

        smoothing = 200 * random.random()  # sigma randomly distributed between 0 annd 200

        print(smoothing)

        # applying gaussian blur to the input image
        output = cv2.bilateralFilter(img, 9, smoothing, smoothing)
        write_output(i, img, output, 'bilateral')
        j += 1
        print('Completed bilateral blur augmentation for image number ' + str(j) + ' of ' + total)

def blurAugmentation(img_lst):
    total = str(len(img_lst))
    j = 0
    for i in img_lst:
        img = cv2.imread(i)

        sigma = random.random() * 2  # sigma randomly distributed between 0 and 2

        # print(sigma)

        # applying gaussian blur to the input image
        output = cv2.GaussianBlur(img, (0, 0), sigma)

        write_output(i, img, output, 'blur')
        j += 1
        print('Completed gaussian blur augmentation for image number ' + str(j) + ' of ' + total)

def motionAugmentation(img_lst):
    total = str(len(img_lst))
    j = 0
    size = 7
    radius = int(size / 2) + 1

    # horizontal motion kernel
    motion_horz = np.zeros((size, size))
    motion_horz[radius, ...] = 1.0 / size

    # diagonal motion kernels
    motion_diag = np.zeros((size, size))
    np.fill_diagonal(motion_diag, 1.0 / size)

    motion_diag_flip = np.flip(motion_diag, 1)  # left/right flip

    for i in img_lst:

        img = cv2.imread(i)

        r = random.randint(0, 3)

        # applying the kernel to the input image
        if r < 2:  # 50% probability of horizontal blur
            output = cv2.filter2D(img, -1, motion_horz)
        if r == 2:
            output = cv2.filter2D(img, -1, motion_diag)
        if r == 3:
            output = cv2.filter2D(img, -1, motion_diag_flip)

        write_output(i, img, output, 'motion')
        print('Completed motion blur augmentation for image number ' + str(j) + ' of ' + total)

def noiseAugmentation(img_lst,maxvar=25,force=False):
    total = str(len(img_lst))
    j=0
    for i in img_lst:

        img = cv2.imread(i)

        # random amount of noise between 0 and 25 (defualt)
        if force == False:
            variance = random.randint(0, maxvar)
        # or if Force is  true forces the max noise
        else:
            variance = maxvar
        # zero mean,  same variance in three image channels
        noise = img.copy()
        cv2.randn(noise, 0, (variance, variance, variance))

        # add noise and clamp to 255
        img = img.astype(int)
        imgPlus = img + noise

        imgPlus[imgPlus > 255] = 255

        img = np.array(img, dtype=np.uint8)
        output = np.array(imgPlus, dtype=np.uint8)

        write_output(i, img, output, 'noise')

        j += 1
        print('Completed noise blur augmentation for image number ' + str(j) + ' of ' + total)

def motionAugmentation_single(img):
    size = 7
    radius = int(size / 2) + 1
    # horizontal motion kernel
    motion_horz = np.zeros((size, size))
    motion_horz[radius, ...] = 1.0 / size

    # diagonal motion kernels
    motion_diag = np.zeros((size, size))
    np.fill_diagonal(motion_diag, 1.0 / size)
    motion_diag_flip = np.flip(motion_diag, 1)  # left/right flip
    r = random.randint(0, 3)
    # applying the kernel to the input image
    if r < 2:  # 50% probability of horizontal blur
        output = cv2.filter2D(img, -1, motion_horz)
    if r == 2:
        output = cv2.filter2D(img, -1, motion_diag)
    if r == 3:
        output = cv2.filter2D(img, -1, motion_diag_flip)
    return(output)

def bilateralAugmentation_single(img):
    smoothing = 200 * random.random()  # sigma randomly distributed between 0 annd 200
    # applying gaussian blur to the input image
    output = cv2.bilateralFilter(img, 9, smoothing, smoothing)
    return(output)

def blurAugmentation_single(img):
    sigma = random.random() * 2  # sigma randomly distributed between 0 and 2
    output = cv2.GaussianBlur(img, (0, 0), sigma)
    return(output)

def noiseAugmentation_single(img,maxvar=25,force=False):
    if force == False:
        variance = random.randint(0, maxvar)
    else:
        variance = maxvar
    # zero mean,  same variance in three image channels
    noise = img.copy()
    cv2.randn(noise, 0, (variance, variance, variance))
    # add noise and clamp to 255
    img = img.astype(int)
    imgPlus = img + noise
    imgPlus[imgPlus > 255] = 255
    output = np.array(imgPlus, dtype=np.uint8)
    return(output)

def get_mask(img):
    #adapted from https://docs.opencv.org/3.0.0/d4/d73/tutorial_py_contours_begin.html
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.focus_class:
        if args.focus_class == 'bicycles':
            ret, thresha = cv2.threshold(imgray, 200, 255, 0) #only cyclist
            ret, threshb = cv2.threshold(imgray, 127, 255, 0) #both bicycle and cylist
            thresh = threshb - thresha
    else:
        ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    return(thresh)

def get_mask_real(img_list,obj='all'):
    total = str(len(img_list))
    img_list.sort()
    for i in range(len(img_list)):
        img_p = img_list[i]
        img = cv2.imread(img_p)
        if obj == 'all':
            ret, thresha = cv2.threshold(img, 23, 255, 0)
            ret, threshb = cv2.threshold(img, 26, 255, cv2.THRESH_BINARY_INV)
            threshc = cv2.bitwise_and(thresha,threshb)
            ret, threshd = cv2.threshold(img, 32, 255, 0)
            ret, threshe = cv2.threshold(img, 33, 255, cv2.THRESH_BINARY_INV)
            threshf = cv2.bitwise_and(threshd, threshe)
            thresh_final = threshc + threshf
        elif obj == 'people_no_riders':
            ret, thresha = cv2.threshold(img, 23, 255, 0)
            ret, threshb = cv2.threshold(img, 24, 255, cv2.THRESH_BINARY_INV)
            thresh_final = cv2.bitwise_and(thresha,threshb)
        elif obj == 'people':
            ret, thresha = cv2.threshold(img, 23, 255, 0)
            ret, threshb = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV)
            thresh_final = cv2.bitwise_and(thresha,threshb)
        elif obj == 'cars':
            ret, thresha = cv2.threshold(img, 25, 255, 0)
            ret, threshb = cv2.threshold(img, 26, 255, cv2.THRESH_BINARY_INV)
            thresh_final = cv2.bitwise_and(thresha,threshb)
        elif obj == 'bicycles':
            ret, thresha = cv2.threshold(img, 32, 255, 0)
            ret, threshb = cv2.threshold(img, 33, 255, cv2.THRESH_BINARY_INV)
            thresh_final = cv2.bitwise_and(thresha,threshb)
        else:
            return Exception('Unknown object specefied. Aborting!')
        write_output(img_p, img, thresh_final, 'Make segmentation mask')
        print('Created make for image number ' + str(i + 1) + ' of ' + total)


def removebackground(img_list,seg_img_list,gaus=False,color=80):
    if len(img_list) != len(seg_img_list):
        raise Exception("Unequal amount of target images, segmentation images. Aborting!")
    total = str(len(img_list))
    img_list.sort()
    seg_img_list.sort()
    for i in range(len(img_list)):
        img_p, seg_img_p = img_list[i],seg_img_list[i]
        img = cv2.imread(img_p)
        img_s = cv2.imread(seg_img_p)
        #get object countours and mask
        mask = get_mask(img_s)
        ret, antimask = cv2.threshold(mask, 200, color, cv2.THRESH_BINARY_INV)
        antimask = cv2.merge((antimask, antimask, antimask))
        #cut out original unblurred image
        result = cv2.bitwise_and(img, img, mask=mask)
        result = result + antimask
        if gaus == True:
            result = blurAugmentation_single(result)
        write_output(img_p,img,result,'Remove background')
        print('Completed remove background for image number ' + str(i+1) + ' of ' + total)

def histogram_renormalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    return(img2)

def blend(foreground, background,alpha):
    foreground = foreground.astype('float32')
    background = background.astype('float32')
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    outImage.astype('uint8')
    return(outImage)

def shrinkAndBlendBackground(img_list,seg_img_list,back_image_list,x_dim,y_dim):
    #if len(img_list) != len(seg_img_list):
    #    raise Exception("Unequal amount of target images, segmentation images. Aborting!")
    total = str(len(img_list))
    img_list.sort()
    seg_img_list.sort()
    for i in range(len(img_list)):
        #get random background image
        b_img_p = back_image_list[random.randrange(len(back_image_list))]
        #get and open foreground image, segmentation image
        img_p, seg_img_p = img_list[i],seg_img_list[i]
        img = cv2.imread(img_p)
        img_o = img
        img_s = cv2.imread(seg_img_p)
        img_b = cv2.imread(b_img_p)
        #blur and shrink objects
        img = shrinkImage(img, x_dim, y_dim)

        #get object countours and mask
        mask = get_mask(img_s)
        mask = mask.astype('float32')
        mask = mask/255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = shrinkImage(mask, x_dim, y_dim, True)
        #resize background image to foreground image size
        img_b = cv2.resize(img_b, (x_dim, y_dim))
        mask = cv2.merge((mask, mask, mask))
        #get object countours and mask
        result = blend(img,img_b,mask)
        write_output(img_p,img_o,result,'Switch background')
        print('Completed blend background for image number ' + str(i+1) + ' of ' + total)

def shrinkImage(img, x_dim,y_dim,blur=False):
    if blur == True:
        img = cv2.GaussianBlur(img, (0, 0), 2)
    img = cv2.resize(img, (x_dim, y_dim), cv2.INTER_AREA)
    return(img)

def resize_images(img_list, x_dim, y_dim):
    total = str(len(img_list))
    i = 0
    for img_p in img_list:
        img_o = cv2.imread(img_p)
        img_deblur = cv2.GaussianBlur(img_o, (0, 0), 2)
        img = cv2.resize(img_deblur, (x_dim, y_dim), cv2.INTER_AREA)
        write_output(img_p, img_o, img, 'resize_images')
        i+=1
        print('Completed resize image for image number ' + str(i) + ' of ' + total)

def switchbackground(img_list,seg_img_list,back_image_list):
    if len(img_list) != len(seg_img_list):
        raise Exception("Unequal amount of target images, segmentation images. Aborting!")
    total = str(len(img_list))
    img_list.sort()
    seg_img_list.sort()
    random.seed(1)
    for i in range(len(img_list)):
        #get random background image
        b_img_p = back_image_list[random.randrange(len(back_image_list))]
        #get and open foreground image, segmentation image
        img_p, seg_img_p = img_list[i],seg_img_list[i]
        img = cv2.imread(img_p)
        img_s = cv2.imread(seg_img_p)
        img_b = cv2.imread(b_img_p)

        #histogram renormalizaiton
        #img = histogram_renormalize(img)
        #resize background image to foreground image size
        d_size = img.shape
        img_b = cv2.resize(img_b, (d_size[1], d_size[0]))
        #get object countours and mask
        mask = get_mask(img_s)
        ret, antimask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
        #antimask = cv2.merge((antimask, antimask, antimask))
        #cut out original unblurred image
        result = cv2.bitwise_and(img, img, mask=mask)
        background = cv2.bitwise_and(img_b, img_b, mask=antimask)
        result = result + background
        write_output(img_p,img,result,'Switch background')
        print('Completed switch background for image number ' + str(i+1) + ' of ' + total)

def backgroundnoise(img_list,seg_img_list,var,gaus=False):
    if len(img_list) != len(seg_img_list):
        raise Exception("Unequal amount of target images, segmentation images. Aborting!")
    total = str(len(img_list))
    img_list.sort()
    seg_img_list.sort()
    for i in range(len(img_list)):
        img_p, seg_img_p = img_list[i],seg_img_list[i]
        img = cv2.imread(img_p)
        img_s = cv2.imread(seg_img_p)
        #blur entire image
        out_image = noiseAugmentation_single(img, var, True)
        #get object countours and mask
        mask = get_mask(img_s)
        ret, antimask = cv2.threshold(mask, 200,255,cv2.THRESH_BINARY_INV)
        #cut out original unblurred image
        clean_object = cv2.bitwise_and(img, img, mask=mask)
        #and remove object from blurred image
        out_image = cv2.bitwise_and(out_image, out_image, mask=antimask)
        #paste object back into blurred image
        result = out_image + clean_object
        #gausian blur whole thing if desired
        if gaus == True:
            result = blurAugmentation_single(result)
        write_output(img_p,img,result,'Background Noise')
        print('Completed background noise augmentation for image number ' + str(i+1) + ' of ' + total)

def write_output(path, img, output, name):
    if show:
        cv2.imshow(name, img)
        cv2.waitKey(0);

        cv2.imshow(name, output)
        cv2.waitKey(0);
    if write:
        if args.output_loc:
            img_name = os.path.basename(path)
            output_name = os.path.join(out_root, img_name)
            cv2.imwrite(output_name, output)
        else:
            cv2.imwrite(path, output)

def s1(img_lst):
    total = str(len(img_lst))
    j = 0
    for i in img_lst:
        img = cv2.imread(i)
        img_inter = blurAugmentation_single(img)
        img_inter = noiseAugmentation_single(img_inter)
        output = bilateralAugmentation_single(img_inter)
        write_output(i,img,output,'s1')
        j += 1
        print('Completed s1 augmentations image number ' + str(j) + ' of ' + total)

def s2(img_lst):
    total = str(len(img_lst))
    j = 0

    for i in img_lst:
        img = cv2.imread(i)
        img_inter = motionAugmentation_single(img)
        img_inter = blurAugmentation_single(img_inter)
        output = noiseAugmentation_single(img_inter)
        write_output(i,img,output,'s2')
        j += 1
        print('Completed s2 augmentations image number ' + str(j) + ' of ' + total)

def s3(img_lst):
    total = str(len(img_lst))
    j = 0
    for i in img_lst:
        img = cv2.imread(i)
        img_inter = motionAugmentation_single(img)
        img_inter = noiseAugmentation_single(img_inter)
        output = bilateralAugmentation_single(img_inter)
        write_output(i,img,output,'s3')
        j += 1
        print('Completed s3 augmentations image number ' + str(j) + ' of ' + total)

def random_sampler():
    os.mkdir(in_root)
    os.mkdir(os.path.join(in_root, 'Data'))
    os.mkdir(os.path.join(in_root, 'Annotations'))
    data_list = sorted(glob.glob(os.path.join(origin_root, 'Data', '*' + exten)))
    anno_list = sorted(glob.glob(os.path.join(origin_root, 'Annotations', '*' + '.json')))
    img_len = len(data_list)
    print('There are a total of ' + str(img_len) + ' images')
    anno_len = len(anno_list)
    if img_len != anno_len:
        shutil.rmtree(in_root)
        raise Exception('Error: Number of images does not equal number of jsons in orign folder, aborrting!')
    if num_file >= img_len:
        shutil.rmtree(in_root)
        raise Exception('Error: Number of files to sample higher than or equal to number files in folder, aborting! Check -o arg!')
    sel_img_numb = random.sample(range(img_len),num_file)
    for ind in sel_img_numb:
        print('Selected image: ' + data_list[ind])
        shutil.copy(data_list[ind], os.path.join(in_root,'Data'))
        shutil.copy(anno_list[ind], os.path.join(in_root, 'Annotations'))
        if not keep:
            os.remove(data_list[ind])
            os.remove(anno_list[ind])

def main():
    if sample:
        random_sampler()
        print('Random Sampling Complete')
        img_lst = glob.glob(os.path.join(in_root, 'Data', '*' + exten))
    else:
        img_lst = glob.glob(os.path.join(in_root,'*'+exten))
        if len(img_lst) == 0:
            raise Exception('Error! No images found aborting! Check path: ' + os.path.join(in_root,'*'+exten))
        if seg == True:
            seg_lst = glob.glob(os.path.join(seg_root, '*'))
            if len(seg_lst) == 0:
                raise Exception('Error! No segmentation images found aborting! Check path: '+ os.path.join(seg_root,'*'+exten))
    if augments == 'gausian' or augments == 'g':
        print('Starting Blur Augmentation')
        blurAugmentation(img_lst)
    elif augments == 'motion' or augments == 'm':
        print('Starting Motion Augmentation')
        motionAugmentation(img_lst)
    elif augments == 'noise' or augments == 'n':
        print('Noise Augmentation')
        noiseAugmentation(img_lst)
    elif augments == 'bilateral' or augments == 'b':
        print('Bilateral Augmentations')
        bilateralAugmentation(img_lst)
    elif augments == 's1':
        s1(img_lst)
    elif augments == 's2':
        s2(img_lst)
    elif augments == 's3':
        s3(img_lst)
    elif augments == 'bn':
        blur = args.blur_amount if args.blur_amount else 25
        backgroundnoise(img_lst,seg_lst,blur)
    elif augments == 'bng':
        blur = args.blur_amount if args.blur_amount else 25
        backgroundnoise(img_lst,seg_lst,blur,True)
    elif augments == 'rb':
        if args.background_color:
            removebackground(img_lst,seg_lst,color=args.background_color)
        else:
            removebackground(img_lst, seg_lst)
    elif augments == 'rbg':
        removebackground(img_lst,seg_lst,True)
    elif augments == 'sb':
        back_lst = glob.glob(os.path.join(background_image_root,'*'))
        switchbackground(img_lst,seg_lst,back_lst)
    elif augments == 'rz':
        resize_images(img_lst, x_dim, y_dim)
    elif augments == 'sh':
        back_lst = glob.glob(os.path.join(background_image_root,'*'))
        shrinkAndBlendBackground(img_lst, seg_lst, back_lst,x_dim, y_dim)
    elif augments == 'mask':
        if args.interest_class:
            get_mask_real(img_lst,args.interest_class)
        else:
            get_mask_real(img_list)
    else:
        raise Exception('Error: unknown augmentation specified, see help for possible augmentations. Aborting!')
    print('Augmentations complete')



if __name__ == '__main__':
    print ("Gaussian noise, motion blur and bilateral filter are migrated to Augmentifier:")
    print ("https://github.com/FLIR/augmentifier/blob/master/examples/all_augmentations.py")
    raise SystemExit

    parser = argparse.ArgumentParser(description="Augment_images")
    parser.add_argument('-i', '--input_root', required=True,
                        help="root to images needing augmentation, these images will be overwritten by default, if -r is true must NOT exist and Annotations and Data folder will be created at the specified path whereas it is assumed you are being pointed to the 'Data' folder wihtout the -r fkag")
    parser.add_argument('-a', '--augmentation',required=True,
                        help='''
                        possible inputs: g or gausian - gausian blur only, n or noise - noise blur only
                        m or motion - motion blur only, b or bilateral - bilateral blur only
                        s1 - gausian -> noise -> bilateral
                        s2 - motion -> gausian -> noise
                        s3 - motion -> noise -> bilateral
                        bn - background noise no noise on objects of interest 
                        bng - same as above with gausian blur afterwards
                        rb - remove background
                        rbg - remove background with gausian blur afterwards 
                        sb - switch background
                        rz - resize image
                        sh - shrink and blend with background
                        mask - convert semantic segmentation into mask - defaults to people, cars, bicycles, rider can be modified with arg -h
                        ''')
    parser.add_argument('-e', '--extension', help="Image extension - defaults to .png")
    parser.add_argument('-s', '--show', action='store_true', help='Creates a pop up of the augmented image, if arg present WILL create pop up otherwise WILL NOT')
    parser.add_argument('-w', '--no_write', action='store_true', help='Writes images, if arg present WILL NOT write otherwise WILL write')
    parser.add_argument('-o', '--output_loc',help="If copies of images with augmentations are desired (instead of overwritting original location) location of outputs")
    #arguments relevant only to random sampling
    parser.add_argument('-r', '--random', action='store_true',help='Sample a random subset of images from another location')
    parser.add_argument('-k', '--keep_original', action='store_true',help='If -r is flagged, this flag will allow you to no delete original images')
    parser.add_argument('-l', '--original_loc',help='Only used if -r is present, location of files to sample from (assumes sort_and_cvt file format')
    parser.add_argument('-n', '--n_files',type=int,help="Number of files to sample")
    #arguments relevant only to background noise and gausin blur on object augmentation
    parser.add_argument('-c', '--segmentation_image_root',help='Path to segmentation images, if required by blur')
    parser.add_argument('-b', '--blur_amount',type=int,help="Amount of blur if using background noise and gausian blur defaults to 25")
    parser.add_argument('-f', '--focus_class',help='Segmentation class to focus on. Currently only arg "bicycles" (ignore rider) is implemented.')
    parser.add_argument('-d', '--background_image_path',help='For switch background augmentation, path to folder containing background images to randomly select from')
    parser.add_argument('-g', '--background_color', type=int,help='Used only for remove background desired background color from 1 (black) to 255 (white) deafults to 80')
    parser.add_argument('-j', '--interest_class', help='''
                        for mask arg allows you to pick what objects to mask possible args:
                        people - people and rider
                        people_no_riders
                        bicycles
                        cars
                        ''')
    #arguments relevant only to image resize
    parser.add_argument('-x', '--x_resize', type=int, help='Resize image in x direction')
    parser.add_argument('-y', '--y_resize', type=int,help='Resize image in y direction')

    args = parser.parse_args()

    in_root = args.input_root
    augments = args.augmentation
    sample = True if args.random else False
    keep = True if args.keep_original else False
    if sample:
        if not args.original_loc or not args.n_files:
            raise Exception('Original loc (-o) and number files (-n) args required if random sampling (arg -r is present)')
        if os.path.exists(in_root):
            raise Exception('If random flag is true (arg -r is present), input root (-i) must not already exist!')
    if args.original_loc: origin_root = args.original_loc
    if args.n_files: num_file = args.n_files
    exten = args.extension if args.extension else '.png'
    show = True if args.show else False
    write = False if args.no_write else True
    seg = False
    if args.output_loc:
        out_root = args.output_loc
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
        os.mkdir(out_root)
    if args.augmentation == 'bn' or args.augmentation == 'bng' or args.augmentation == 'rb' or\
            args.augmentation == 'rbg' or args.augmentation == 'sb' or args.augmentation == 'sh':
        seg = True
        seg_root = args.segmentation_image_root
    if args.augmentation == 'sb' or args.augmentation == 'sh':
        background_image_root = args.background_image_path
    if augments == 'rz' or args.augmentation == 'sh':
        x_dim = args.x_resize
        y_dim = args.y_resize
    main()

