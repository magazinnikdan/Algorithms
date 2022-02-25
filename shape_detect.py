# import cv2
# import numpy as np
# img = cv2.imread('shape5.jpg')  #read image from system
# cv2.imshow('original', img)    #Displaying original image
# cv2.waitKey(0)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #Convert to grayscale image
# edged = cv2.Canny(gray, 170, 255)            #Determine edges of objects in an image
#
#
#
# mean = 0.0   # some constant
# std = 2.0    # some constant (standard deviation)
# noisy_img = img + np.random.normal(mean, std, img.shape)
# noisy_img_clipped = np.clip(noisy_img, 0, 255)
# cv2.imshow('noisy', noisy_img_clipped)
# cv2.waitKey(0)
# print(noisy_img_clipped)
#
# ret,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
#
# (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours in an image
# def detectShape(c):          #Function to determine type of polygon on basis of number of sides
#        shape = 'unknown'
#        peri=cv2.arcLength(cnt,True)
#        vertices = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#        sides = len(vertices)
#        if (sides == 3):
#             shape='triangle'
#        elif(sides==4):
#              x,y,w,h=cv2.boundingRect(cnt)
#              aspectratio=float(w)/h
#              if (aspectratio==1):
#                    shape='square'
#              else:
#                    shape="rectangle"
#        elif(sides==5):
#             shape='pentagon'
#        elif(sides==6):
#             shape='hexagon'
#        elif(sides==8):
#             shape='octagon'
#        elif(sides==10):
#             shape='star'
#        else:
#            shape='circle'
#        return shape
# for cnt in contours:
#     moment=cv2.moments(cnt)
#     cx = int(moment['m10'] / moment['m00'])
#     cy = int(moment['m01'] / moment['m00'])
#     shape=detectShape(cnt)
#     cv2.drawContours(img,[cnt],-1,(0,255,0),2)
#     cv2.putText(img,shape,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)  #Putting name of polygon along with the shape
#     cv2.imshow('polygons_detected',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
################################################################################################################
# from __future__ import print_function
# import cv2 as cv
# import numpy as np
# import argparse
# import random as rng
# import matplotlib.pyplot as plt
# rng.seed(12345)
#
#
# def thresh_callback(val):
#     threshold = val
#
#     canny_output = cv.Canny(src_gray, threshold, threshold * 2)
#
#     _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     contours_poly = [None] * len(contours)
#     boundRect = [None] * len(contours)
#     centers = [None] * len(contours)
#     radius = [None] * len(contours)
#     for i, c in enumerate(contours):
#         contours_poly[i] = cv.approxPolyDP(c, 3, True)
#         boundRect[i] = cv.boundingRect(contours_poly[i])
#         centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
#
#     drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
#
#     for i in range(len(contours)):
#         color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
#         cv.drawContours(drawing, contours_poly, i, color)
#         cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
#         cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
#
#     cv.imshow('Contours', drawing)
#
#
# parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
# args = parser.parse_args()
# # src = cv.imread(cv.samples.findFile(args.input))
# src = cv.imread('shape5.jpg')
# edges = cv.Canny(src,100,200)
#
# plt.subplot(121),plt.imshow(src,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# # Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.blur(src_gray, (3, 3))
# source_window = 'Source'
# cv.namedWindow(source_window)
# cv.imshow(source_window, src)
# max_thresh = 200
# thresh = 50  # initial threshold
# cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)
# cv.waitKey()


##############################################################################################################
import cv2
def show_image(image):
    cv2.imshow('image',image)
    c = cv2.waitKey()
    if c >= 0 : return -1
    return 0
image = cv2.imread('shape6.jpeg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(im, contours, -1, (0,255,75), 2)
show_image(img)