# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:59:49 2018

@author: z003vrzk
"""

import matplotlib.pyplot as plt
import numpy as np

a1 = 0.7
#b1 = 0.6
c1 = 0.03

fig = plt.figure(1)
plt.title('Exponential Fits')
plt.xlabel('Distance between node & pixel (pixels)')
plt.ylabel('Magnitude')

x1 = np.arange(0,100,0.5)
for b1 in np.arange(0.1,0.9,0.1):
    y1 = a1**x1**b1 + c1
    plt.plot(x1,y1,lw=2, label = 'b1 = ' + str(round(b1,1)))

    
plt.legend()
plt.waitforbuttonpress(0) & 0xff
plt.close(fig)
