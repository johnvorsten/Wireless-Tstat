"""
Created on Wed Aug 29 12:54:30 2018
@author: z003vrzk
"""
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

def importData():
#    global data
    global cal
    data = pd.ExcelFile('JVImports\Calibration Data.xlsx') #Load in temp data
    sheetName = data.sheet_names
    calibrationData = data.parse(sheetName[0]) #Create data frame
    cal = calibrationData
    cal = cal.rename(columns = {'Unnamed: 12' : 'Thermistor8'})
    
#Accessing pandas objects:
#    I can access by using dataframe.loc[rowindex, 'Column index or label'] - will return item in row & column address
#    To access a whole column: dataframe.loc[:,'Column labl']
#    To access a whole row: dataframe.loc[row index] ex cal.loc[5]
#    To access multiple columsn: dataframe.loc['column1', 'column2', 'column3']
#    To slice multiple rows: dataframe.loc[rowIndex1:rowIndex2] ex cal.loc[1:5]
#    To slice rows another way: cal[:5]
#    To slice rows and columns: dataframe.loc[row index, :'Column label']
    
def fixData():
    global error
    global cal
#    global error_temp
    trueTemp = 79.19
    cal = cal.drop(index = cal.index[0:10])
    cal = cal.drop(index = 126)
    cal = cal.drop(index = 229)
    cal.reindex(range(0,cal.shape[0]))
    error_temp = np.array(trueTemp - cal.loc[:, 'Thermistor0':]) #add this to the normal data
    error = np.zeros((1,error_temp.shape[1]))
    
    for i in range(0,error_temp.shape[1]):
        error[0,i] = stat.mean(error_temp[:,i])
    return error
        
def main():
    importData()
    fixData()
    
def var():
    std = np.zeros(2468)
    diff = np.zeros(2468)
    for i in range(0, 2467):
#        Var = stat.variance(tData[i, :])
#        print('Variance = ', round(Var, 2))
        data = np.append(tData[i, 1], tData[i, 3:])
        std[i] = stat.stdev(data)
        diff[i] = np.ptp(data)
    x = np.arange(0, 2468)
    y = std

    plt.subplot(2,1,1)
    plt.plot(x,y)
    plt.xlabel('Measurement iteration')
    plt.ylabel('Standard deviation (Deg F)')
    plt.title('Standard Deviation Across Measurements')
    
    plt.subplot(2,1,2)
    plt.plot(x, diff)
    plt.xlabel('Measurement iteration')
    plt.ylabel('Range')
    
    