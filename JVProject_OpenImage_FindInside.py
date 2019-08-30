# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:07:20 2018

@author: z003vrzk
"""

import numpy as np
import cv2
import JVProjectModule as JV
import math

#read in the image
img1 = cv2.imread('Images\Apartment_Bare.png')
img2 = cv2.imread('Images\Apartment_Detail.png')

#resize the image
img1 = cv2.resize(img1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

#selec the ROI
#roi = cv2.selectROI(img1)
#hard code the image size so you dont lose it
x1, y1, x2, y2 = 200, 36, 496, 525

img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#crop the image... selectROI gives the distance moved not the actual coordinate
img1crop = img1Gray[y1:y2+y1,x1:x2+x1]

#check if the pixel is inside the apartment
#when using np.shape.. the first is the number of rows (the y)
#and the second is the number of columns (the x)



for h in  range(0,np.shape(img1crop)[0]):
    #h is the value of the current pixel in question y value
    for i in range(0,np.shape(img1crop)[1]): #do over whole image
        #i is the current x dimension we are calculating rays for
        for j in range(i,np.shape(img1crop)[1]): #calculate the rays for each pixel
            #j is the value of the ray x value
            for k in range(0,3):
                #k is the trionometric angle.. do 0pi/3, 1pi/3, 2pi/3
                x = j - i #this is the x value of the ray specific pixel
                y = math.ceil(x*math.tan(math.pi*k/3) + h) #this is the y valve of the ray specific pixel
                print('y value = ', y, 'x value = ', x, '\n')
                JV.isBlack(img1crop[y,x])
                


cv2.imshow('Bare Apartment image', img1)
cv2.imshow('Cropped Image', img1crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
