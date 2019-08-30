# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:28:04 2018
@author: z003vrzk
"""
import os
import numpy as np
import cv2
import math
import pandas as pd
import JVProject_Calibration
import time

def main():
    global nodes, loopIndex
    global error, tData, x1, x2
    importImages()
    resizeImages()
    convertToGray()
    getMask()
    #getNodes()
    nodes = np.array([[139, 390, 0],
             [288, 417, 1],
#             [307, 464, 2],
             [419, 379, 3],
             [214, 301, 4],
             [310, 257, 5],
             [441, 201, 6],
             [277, 189, 7],
             [121, 213, 8]]) #Pixel y, x, node ID
    getPixelWeightArray()
    importData()
    JVProject_Calibration.importData() #get the data
    error = JVProject_Calibration.fixData() #get the error matrix which i need
    calculateDistance2()
    a = np.where(calc_temp[:,:,0] > 0)
    loopIndex = np.transpose(np.array((a[0], a[1]))) #in the future put this outside so it only runs once
    tData = np.array(df1.loc[:, 'Thermistor0':])
    
    b1 = np.array(([120], [0]))
    A1 = np.array(([70,1],[85,1]))
    x1 = np.linalg.solve(A1,b1) #Hue = x1[0]*temp + x1[1] when temp < 85
    
    b2 = np.array(([20],[0]))
    A2 = np.array(([85,1],[110,1]))
    x2 = np.linalg.solve(A2, b2) #Hue = x2[0]*temp + x2[1] when temp > 85
    

    
    

def importImages():
    global img1
    global img2
    #read in the image
    cwd = os.getcwd()
    #os.chdir(os.path.join(cwd, '.spyder-py3\Scripts'))
    img1 = cv2.imread(os.path.join(cwd, 'JVImports\Images\Apartment_Bare.png'))
    img2 = cv2.imread(os.path.join(cwd, 'JVImports\Images\Apartment_Detail.png'))
    #img1 = cv2.imread('JVImports\Images\Apartment_Bare.png')
    #img2 = cv2.imread('JVImports\Images\Apartment_Detail.png')

def resizeImages():
    global img1
    global img2
    #resize the image
    img1 = cv2.resize(img1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #selec the ROI
    #roi = cv2.selectROI(img1)
    #hard code the image size so you dont lose it
    x1, y1, x2, y2 = 200, 36, 496, 525
    ##crop the image... selectROI gives the distance moved not the actual coordinate
    img1 = img1[y1:y2+y1,x1:x2+x1]
    img2 = img2[y1:y2+y1,x1:x2+x1]


def convertToGray():
    #Convert the image to Black/white
    global img1_gray
    global img1_inv
    ret, img1_gray = cv2.threshold(img1,220,255,0)
    img1_gray = cv2.cvtColor(img1_gray, cv2.COLOR_BGR2GRAY)
    img1_inv = cv2.bitwise_not(img1_gray)

def getMask():
    global img_contour, img2
    global img1_inv
    global img2_gray
    global img2_mask
    global temperature_mask
    #Find the locations inside the walls
    img_contour, contours, heirarchy = cv2.findContours(img1_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #draw the contours, send thickness = cv2.FILLED
    #img_contour = cv2.drawContours(img_contour, contours, -1, [0,255,255], thickness=cv2.FILLED)
    img_contour = cv2.drawContours(img1, contours, 3, [0,0,0], -1)
    
    #This will be the binary representation of the areas we will want to color in
    #once i get the temperature data
    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
    ret, img1_inv = cv2.threshold(img1_inv, 220, 255, 0)
    res = cv2.bitwise_or(img_contour, img1_inv)
    temperature_mask = cv2.bitwise_not(res) #this is waht i want
    
    #now to subtract the areas that have important information like the couch
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, img2 = cv2.threshold(img2, 245, 255, cv2.THRESH_BINARY)
    ret, img2_mask = cv2.threshold(img2_gray, 220, 255, 0)
    #now to remove the detailed walls
    img2_mask = cv2.bitwise_or(img_contour, img2_mask)
    
    #~~~~this is the grand shebang ill use bitwise to find where to color in ~~~~~~~~~~~
    img2_mask = cv2.bitwise_and(temperature_mask, img2_mask)
    img2_mask[np.where((img2_mask >= 1) & (img2_mask < 255))] = 0


def get_nodes(img):
    global nodes
    nodes = np.zeros((9,2), int)
    for i in range(0,9):
        ROI = cv2.selectROI('img', img)
        a = ROI[0] + ROI[2]//2
        b = ROI[1] + ROI[3]//2
        nodes[i,:] = [b,a] #First coordinate is x, second is y
        cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
    print('Here are your nodes :\n', nodes)
    print('Format : [y coordinate, x coordinate]')
    
def getPixelWeightArray():
    global calc_temp
    global mask
    #Find each pixels relative node
    #def find_node(image, image mask):
#    calc_temp = np.zeros((np.where(img2_mask > 0)[0].size, 8)) #pixels that will be colored in
    calc_temp = np.zeros((img2_mask.shape[0], img2_mask.shape[1], 11))
    calc_temp[:, :, 0] = img2_mask
    mask = calc_temp[:,:,0] > 1
    calc_temp[:,:,9] = 255 #Saturation and Value (HSV Color)
    calc_temp[:,:,10] = 255*mask #Value, make it black with the mask where its not in the apartment
    #def find_pixel(): #find all white pixels in the mask
#    k = 0
    #    global calc_temp
#    for i in range(0, img2_mask.shape[0]): #y coordinate
#        for j in range(0,img2_mask.shape[1]): #x coordinate
#            if img2_mask[i,j] > 0: #if the pixel is not black
#                calc_temp[k,0:2] = [i,j]
#                k = k + 1
    
def importData():
    global data
    global df1
    data = pd.ExcelFile('JVImports\Collected Time Temp.xlsx') #Load in temp data
    sheetName = data.sheet_names
    df1 = data.parse(sheetName[0]) #Create data frame
    #next do data calibration
    #change the data frame to an numpy array
    #df1values = np.float32(df1.values)
    #df1Column = df1.columns
    #df1Index = df1.index
                
def calculateDistance():
    global calc_temp
    distance = np.zeros((nodes.shape[0],2))
    
    for i in range(0,error.shape[1]): #calibrate the temp sensors with my error data
        df1.loc[:,df1.columns[4+  i]] = df1.loc[:, df1.columns[4 + i]] + error[0,i]
    
    
    for i in range(0,calc_temp.shape[0]): #loop through each row (y coor)
        for j in range(0, calc_temp.shape[1]):
            if calc_temp[i, j, 0] > 0:
                for k in range(0,distance.shape[0]): #loop through column (x coor)
        
                    distance[k,0] = math.sqrt( (i - nodes[k,0])**2 + (j - nodes[k,1])**2)
                    distance[k,1] = nodes[k,2] #hold the Node ID
        
                ind = np.argsort( distance[:,0]) #sort by the first column
                distance = distance[ind] #sort by the first column
                
                a = distance[0,0] #pixel to closest node
                b = distance[1,0] #pixel to second closest node
        
                y0 = nodes[nodes[:,2]==distance[0,1], 0] #looking up Nodes based on ID
                y1 = nodes[nodes[:,2]==distance[1,1], 0]
                x0 = nodes[nodes[:,2]==distance[0,1], 1]
                x1 = nodes[nodes[:,2]==distance[1,1], 1]
                c = math.sqrt((y0 - y1)**2 + (x0 - x1)**2) #distance between nodes
                
                try:
                    gamma = math.acos((a**2 + b**2 - c**2)/(2*a*b))/math.pi*180 #law of cosines
                except ValueError:
                    print((a**2 + b**2 - c**2)/(2*a*b))
                    print('Invalid Indicy', i)
                
                dist1 = distance[0,0] #distance 1 
                dist2 = distance[1,0] #distance 2
                dist3 = distance[2,0] #distance 3
                
                if gamma < 135: #decide if we include the third closest distance node
                    mag3 = distance[2,0] #distance 3
                else:
                    mag3 = 0

                tot_dist = dist1 + dist2 + dist3
                #calculate magnitudes for each node
                calc_temp[i, j, 1] = dist1/tot_dist #Reassign the mag to 1
                calc_temp[i, j, 3] = dist2/tot_dist #reassign second mag to 2
                calc_temp[i, j, 5] = dist3/tot_dist #reasssign third mag to 3
                #Keep node indicy
                calc_temp[i, j, 2] = distance[0,1]
                calc_temp[i, j, 4] = distance[1,1]
                calc_temp[i, j, 6] = distance[2,1]
            else:
                continue
            
def calculateDistance2(): #use a different weighting scheme - exponential instead of linear weight
    global calc_temp
    a = 0.7 #change this to affect how quickly it decreases as it goes farther away
    b = 0.4 #also affects how quickly it decreases as well as the inital slope
    distance = np.zeros((nodes.shape[0],2))
    
    for i in range(0,error.shape[1]): #calibrate the temp sensors with my error data
        df1.loc[:,df1.columns[4+  i]] = df1.loc[:, df1.columns[4 + i]] + error[0,i]
    
    
    for i in range(0,calc_temp.shape[0]): #loop through each row (y coor)
        for j in range(0, calc_temp.shape[1]):
            if calc_temp[i, j, 0] > 0:
                for k in range(0,distance.shape[0]): #loop through column (x coor)
        
                    distance[k,0] = math.sqrt( (i - nodes[k,0])**2 + (j - nodes[k,1])**2)
                    distance[k,1] = nodes[k,2] #hold the Node ID
        
                ind = np.argsort( distance[:,0]) #sort by the first column
                distance = distance[ind] #sort by the first column
                
#                a = distance[0,0] #pixel to closest node
#                b = distance[1,0] #pixel to second closest node
#        
#                y0 = nodes[nodes[:,2]==distance[0,1], 0] #looking up Nodes based on ID
#                y1 = nodes[nodes[:,2]==distance[1,1], 0]
#                x0 = nodes[nodes[:,2]==distance[0,1], 1]
#                x1 = nodes[nodes[:,2]==distance[1,1], 1]
#                c = math.sqrt((y0 - y1)**2 + (x0 - x1)**2) #distance between nodes
#                
#                try:
#                    gamma = math.acos((a**2 + b**2 - c**2)/(2*a*b))/math.pi*180 #law of cosines
#                except ValueError:
#                    print((a**2 + b**2 - c**2)/(2*a*b))
#                    print('Invalid Indicy', i)
                
#                dist1 = distance[0,0] #distance 1 
#                dist2 = distance[1,0] #distance 2
#                dist3 = distance[2,0] #distance 3
                #weight the distances effects based on an exponential type function
                mag1 = a**distance[0,0]**b
                mag2 = a**distance[1,0]**b
                mag3 = a**distance[2,0]**b

                tot_mag = mag1 + mag2 + mag3
                #calculate magnitudes for each node
                calc_temp[i, j, 1] = mag1/tot_mag #Reassign the first mag to 1
                calc_temp[i, j, 3] = mag2/tot_mag #reassign second mag to 2
                calc_temp[i, j, 5] = mag3/tot_mag #reasssign third mag to 3
                #Keep node indicy
                calc_temp[i, j, 2] = distance[0,1] #hold the node identifier
                calc_temp[i, j, 4] = distance[1,1] #hold the node identifier
                calc_temp[i, j, 6] = distance[2,1] #hold the node identifier
            else:
                continue
            
def calculateTemperature2(dataTime): #time is an iteration of the df1 array

#    for pix in loopIndex:
        #Retrieve nodes
#        n1 = int(calc_temp[2, pix[0], pix[1]]) #corresponding node 1 for pixel in calc_temp array
#        n2 = int(calc_temp[4, pix[0], pix[1]]) #etc
#        n3 = int(calc_temp[6, pix[0], pix[1]])
            
            #lookup corresponding node temperatures
#        t1 =  tData[dataTime, int(calc_temp[2, pix[0], pix[1]])]
#        t2 =  tData[dataTime, int(calc_temp[4, pix[0], pix[1]])]
#        t3 =  tData[dataTime, int(calc_temp[6, pix[0], pix[1]])]
            
#        #Calculate temperature by applying weights
#        temperature = (calc_temp[pix[0], pix[1], 1]*tData[dataTime, int(calc_temp[pix[0], pix[1], 2])] + 
#            calc_temp[pix[0], pix[1], 3]*tData[dataTime, int(calc_temp[pix[0], pix[1], 4])] + 
#            calc_temp[pix[0], pix[1], 5]*tData[dataTime, int(calc_temp[pix[0], pix[1], 6])])
#        print('temperature', temperature)
#        calc_temp[pix[0], pix[1], 7] = temperature

    #Calculate pixel temp by multiplying node temperature by pixel magnitude
    calc_temp[:,:,7] = (tData[dataTime, calc_temp[:,:,2].astype(np.uint8)]*calc_temp[:,:,1] #node temperatuer array * pixel magnitude array
        + tData[dataTime, calc_temp[:,:,4].astype(np.uint8)]*calc_temp[:,:,3]
        + tData[dataTime, calc_temp[:,:,6].astype(np.uint8)]*calc_temp[:,:,5])
        
    #Map temperature to color, these will be the only pixels that change
    calc_temp[:,:,8] = (calc_temp[:,:,7]*x1[0] + x1[1])*(mask)

test = calc_temp[:,:,8:]
test = test.astype(np.uint8)
test = cv2.cvtColor(test, cv2.COLOR_HSV2BGR)

cv2.imshow('image 2 mask, img2_mask', img2_mask)
cv2.imshow('test', test)
cv2.imshow('image 2, img2', img2)
cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()
