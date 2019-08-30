# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:38:52 2018

@author: z003vrzk
"""

#Vt = V0*Rt/(R1 + Rt)
#Rt = Vt*R1/(Vo-Vt)

#I want a graph of resistance v temperature and 
#resistance change v temperature
#also resistance v voltage
#and temperature v voltage
import matplotlib.pyplot as plt
import numpy as np

resistanceArray = np.array([28365, 26834, 25395, 24042, 22770, 21573, 20446, 19376, 18378, 17437, 16550, 15714, 14925, 14180 , 13478, 12814 , 12182, 11590, 11030 , 10501 , 10000, 9526  , 9078 , 8653 , 8251 , 7866, 7505, 7163, 6838, 6530, 6238, 5960 , 5697, 5447 , 5207,  4981,  4766,  4561])
tempArray = np.array([37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111])
V0 = 5
R1 = 10000

def calcResistance(Vt, R1, V0):
    R_thermistor = Vt*R1/(V0-Vt)
    return R_thermistor

def calcVoltage(V0, R1, R_thermistor,):
    V_thermistor = V0*R_thermistor/(R1 + R_thermistor)
    return V_thermistor

def calcTemp(R_thermistor, Res_Array, Temp_Array):
    i = 0
    if R_thermistor >= Res_Array[0]:
        print('thermistor resistance too high')
        return Temp_Array[0]
    if R_thermistor <= Res_Array[Res_Array.shape[0]-1]:
        print('thermistor resistance too low')
        return Temp_Array[Res_Array.shape[0]-1]
    while R_thermistor < Resistance_Array[i]:
        i = i + 1
    measuredTemp = (Temp_Array[i] - Temp_Array[i-1])/(Res_Array[i]-Res_Array[i-1])*(R_thermistor - Res_Array[i]) + Temp_Array[i-1]
    
    return measuredTemp
    

#Graph of voltage versus resistance
thermistorVoltage = np.zeros((resistanceArray.shape[0]))
measuredTemp = np.zeros((resistanceArray.shape[0]))
for i in range(0, resistanceArray.shape[0]):
    thermistorVoltage[i] = calcVoltage(V0, R1, resistanceArray[i])
    measuredTemp[i] = calcTemp(resistanceArray[i], resistanceArray, tempArray)

testTemp = np.zeros((2**10-1))
for i in range(0, 2**10-1):
    testResistance = calcResistance(i*5/(2**10-1), R1, V0)
    testTemp[i] = calcTemp(testResistance, resistanceArray, tempArray)
#testTemp = np.delete(testTemp, np.where(testTemp == 111))
#testTemp = np.delete(testTemp, np.where(testTemp == 37))
    
fig = plt.figure(1)
host = fig.add_subplot(111)

par1 = host.twinx() #Temperature
#par2 = host.twinx()

host.set_xlim(resistanceArray[0], resistanceArray[resistanceArray.size-1])
host.set_ylim(0,5)
par1.set_ylim(0,100)

host.set_xlabel('Thermistor Resistance')
host.set_ylabel('Thermistor measured Voltage')
par1.set_ylabel('Measured Temperature')

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)

p1, = host.plot(resistanceArray, thermistorVoltage, color = color1, label = 'Thermistor Voltage')
p2, = par1.plot(resistanceArray, measuredTemp, color = color2, label = 'Measured Temperature')

plt.title('Resistance versus measured voltage')

fig2 = plt.figure(2)
plt.plot(np.arange(0,2**10-1)*(5/1023), testTemp)

plt.waitforbuttonpress(0)
plt.close()    