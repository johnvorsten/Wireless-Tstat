# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 12:21:15 2018

@author: z003vrzk
"""

import os
import pandas as pd
import numpy as np
import math

#current_dir = os.getcwd()
#fileName = 'Collected Temp Data.xlsx'
#data = pd.ExcelFile(os.path.join(current_dir, 'JVImports\Collected Temp Data.xlsx'))
data = pd.ExcelFile('JVImports\Collected Time Temp.xlsx')

#print sheet names
sheetName = data.sheet_names

#load the sheet into a data frame
df1 = data.parse(sheetName[0])

#change the data frame to an numpy array
df1values = np.float32(df1.values)
df1Column = df1.columns
df1Index = df1.index

#fix time, create time keeping numpy arrays
#hour = np.zeros(time.size); minute = np.zeros(time.size); second = np.zeros(time.size)
#secondElapsed = np.zeros(time.size)

#def first_n_digits(num,n): #claculate the first n digits of an integer
##    digits = num//(10**(int(math.log10(num)) -n + 1))
#    a = str(num)
#    return int(a[0:n])
#
#def last_n_digits(num,n): #calculate the last n digits of an integer
##    return num - (num//(10**(int(math.log10(num))-n))*(10**n))
#    a = str(num)
#    return int(a[len(a)-n:len(a)])
#
##my data originally had the weekday included.. need to remove the weekday
##this mehtod doesnt retain leading zeros...
#for i in range(4118,len(time)): #need to fix the whole date/time thing
#    time[i] = last_n_digits(time[i], int(math.log10(time[i])))
#
#for i in range(0,time.size):
#    if int(math.log10(time[i])) + 1 == 6:
#        hour[i] = first_n_digits(time[i],2) #store hour
#        minute[i] = first_n_digits(last_n_digits(time[i],4),2) #store minute
#        second[i] = last_n_digits(time[i],2) #store second
#        secondElapsed[i] = (hour[i]*(60**2) + 60*minute[i] + second[i])
#    elif (int(math.log10(time[i])) + 1) <= 2:
#        secondElapsed[i] = time[i]
#    else:
#        secondElapsed[i] = secondElapsed[i-1] + 10
#        hour[i] = secondElapsed[i]//(60**2)
#        minute[i] = (secondElapsed[i]-hour[i]*(60**2))//60
#        second[i] = (secondElapsed[i] - hour[i]*(60**2) - minute[i]*60)
#        
#
##create a dataframe for the new information, and add in temperature
#DataTime = np.stack((secondElapsed, hour, minute, second), axis=1)
#del time
##del temporaryTime
#del second
#del minute
#del hour
#del secondElapsed
#temperatureDF = df1.loc[:, 'Thermistor0':'Thermistor8']
#DataTemperature = temperatureDF.values
#del temperatureDF
#del df1
#del df1values
#del i



