'''
Q1'2018 : from online - mod'ed a little for experimentation
        : Import blurRoi as a module to use these
'''

import cv2
import numpy as np

def SoftBlurContours(image, contours, ksize, sigmaX, *args, **kwargs):
    iterations = 3
    if 'iters' in kwargs:
	iterations = kwargs['iters']
    sigmaY = args[0] if len(args) > 0 and args[0] != None else sigmaX
    mksize = args[1] if len(args) > 1 and args[1] != None else ksize
    msigmaX = args[2] if len(args) > 2 and args[2] != None else sigmaX
    msigmaY = args[3] if len(args) > 3 and args[3] != None else msigmaX
    mask = np.zeros(image.shape[:2])
    for i, contour in enumerate(contours):
	cv2.drawContours(mask, contour, 0, 255, -1)
    blurred_image = cv2.GaussianBlur(image, (ksize,ksize), sigmaX, None, sigmaY)
    result = np.copy(image)
    for _ in xrange(iterations):
	alpha = mask/255.
	result = alpha[:, :, None]*blurred_image + (1-alpha)[:, :, None]*result
	mask = cv2.GaussianBlur(mask, (mksize, mksize), msigmaX, None, msigmaY)
    return result

def SoftBlurRect(image, rect, ksize, sigmaX, *args, **kwargs):
    x,y,w,h = rect
    contours = [[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]]
    return SoftBlurContours(image, contours, ksize, sigmaX, *args, **kwargs)

def BlurContours(image, contours, ksize, sigmaX, *args):
    sigmaY = args[0] if len(args) > 0 else sigmaX
    mask = np.zeros(image.shape[:2])
    for i, contour in enumerate(contours):
	cv2.drawContours(mask, contour, -1, 255, -1)
	#cv2.drawContours(mask, contour, i, 255, -1)
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX, None, sigmaY)
    result = np.copy(image)
    alpha = mask/255.
    result = alpha[:, :, None]*blurred_image + (1-alpha)[:, :, None]*result
    return result

def BlurRect(image, rect, ksize, sigmaX, *args):
    x,y,w,h = rect
    contours = [[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]]
    return BlurContours(image, contours, ksize, sigmaX, *args)

def main():
    img = cv2.imread("blurInput.jpg")
    #out1 = BlurRect(img, [597,285,81,45], 5, 5)  # float doesn't work. use this for bbox blur
    #out3 = SoftBlurRect(img, [597,285,81,45], 5, 5)
    #out5 = SoftBlurRect(img, [597,285,81,45], 27, 5, 5, 5, iters=2)
    #out2 = BlurContours(img, [[np.array([[500,50],[600,300],[150,200]])]], 17, 5)
    x,y,w,h = 597, 285, 81,45 
    contours = [[np.zeros((4,2), dtype=int)]]*2
    #contours = [[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])], [np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]]
    contours[0] = [np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]
    x += 100
    y += 50
    contours[1] = [np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]
    out4 = BlurContours(img, contours, 5, 5)
    #out4 = SoftBlurContours(img, [[np.array([[500,50],[600,300],[150,200]])],[np.array([[800,500],[950,500],[900,650]])]], 17, 5)

    #cv2.imwrite('harsh-rect.jpg', out1)
    #cv2.imwrite('soft-rect.jpg', out3)
    #cv2.imwrite('soft-rect-args.jpg', out5)
    #cv2.imwrite('harsh-contour.jpg', out2)
    cv2.imwrite('soft-contour.jpg', out4)


if __name__ == '__main__':
    main()


