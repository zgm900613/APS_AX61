# -*- coding: utf-8 -*-
"""
@author: GM Zhu
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

font = {'family': 'Arial'}
ind = np.arange(10)

def in_Excel(file_path):
    data_file = pd.read_excel(file_path, skip = 0)
    dataF = data_file.ix[:, :]
    data_matrix = np.matrix(dataF)
    return(data_file, dataF, data_matrix)

file_names = os.listdir(r'E:\SynchrotronXray\APS20180423\XRDresults\AX61-10degs\Excel')
file_path1 = 'E:/SynchrotronXray/APS20180423/XRDresults/AX61-10degs/Excel/'

data_list = []
for i in tqdm(range(len(file_names))):
    file_path = file_path1 + file_names[i]
    data_file, data0F, data0 = in_Excel(file_path)
    data_list.append(data0)

Lat_str = np.zeros((86, 14, 10))
xxx, yyy, zzz = np.shape(Lat_str)

plane_name = ('(10-10)', '(0002)', '(10-11)', '(10-12)', '(11-20)', 
              '(10-13)', '(20-20)', '(11-22)', '(20-21)', 
              '(220)', '(222)', '(422)', '(511)', '(440)')

for i in tqdm(range(xxx)):
    for j in range(yyy-5):
        for k in range(zzz):
            Lat_str[i, j, k] = data_list[k][i+5, 10*j+17]
    for j in range(5):
        for k in range(zzz):
            Lat_str[i, j+9, k] = data_list[k][i+5, 10*j+110]

plt.figure(figsize = (15, 15))

'''------------------------------------------------------------------------------'''

plt.subplot(2, 2, 1)
step1 = 12 #0.2%
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40, label = plane_name[0])
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40, label = plane_name[2])
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40, label = plane_name[3])
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40, label = plane_name[4])
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[5])
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40, label = plane_name[7])
    else:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[step1, 0, :],c='b')
plt.plot(range(10), Lat_str[step1, 2, :],c='r')
plt.plot(range(10), Lat_str[step1, 3, :],c='g')
plt.plot(range(10), Lat_str[step1, 4, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 5, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[step1, 7, :],c='m')

plt.text(5.2, 0.009, '0.2%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(2, 2, 2)
step1 = 24 #0.2%
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40, label = plane_name[0])
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40, label = plane_name[2])
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40, label = plane_name[3])
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40, label = plane_name[4])
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[5])
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40, label = plane_name[7])
    else:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[step1, 0, :],c='b')
plt.plot(range(10), Lat_str[step1, 2, :],c='r')
plt.plot(range(10), Lat_str[step1, 3, :],c='g')
plt.plot(range(10), Lat_str[step1, 4, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 5, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[step1, 7, :],c='m')

plt.text(5.2, 0.009, '2.0%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(2, 2, 3)
step1 = 52 #0.2%
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40, label = plane_name[0])
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40, label = plane_name[2])
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40, label = plane_name[3])
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40, label = plane_name[4])
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[5])
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40, label = plane_name[7])
    else:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[step1, 0, :],c='b')
plt.plot(range(10), Lat_str[step1, 2, :],c='r')
plt.plot(range(10), Lat_str[step1, 3, :],c='g')
plt.plot(range(10), Lat_str[step1, 4, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 5, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[step1, 7, :],c='m')

plt.text(5.5, 0.009, '10.0%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(2, 2, 4)
step1 = 83 #0.2%
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40, label = plane_name[0])
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40, label = plane_name[2])
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40, label = plane_name[3])
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40, label = plane_name[4])
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[5])
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40, label = plane_name[7])
    else:
        plt.scatter(i, Lat_str[step1, 0, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 2, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[step1, 3, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 4, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 5, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[step1, 7, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[step1, 0, :],c='b')
plt.plot(range(10), Lat_str[step1, 2, :],c='r')
plt.plot(range(10), Lat_str[step1, 3, :],c='g')
plt.plot(range(10), Lat_str[step1, 4, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 5, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[step1, 7, :],c='m')

plt.text(5.5, 0.009, '25.0%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.savefig('test.jpg',dpi = 300, bbox_inches = 'tight')


















plt.figure(figsize = (15, 15))

'''------------------------------------------------------------------------------'''

plt.subplot(2, 2, 1)
step1 = 12 #0.2%
for i in range(zzz):
    if i == 0:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40, label = plane_name[9])
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40, label = plane_name[10])
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40, label = plane_name[11])
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40, label = plane_name[12])
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[13])
    else:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40)
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40)
#lines between points
#plt.plot(range(10), Lat_str[step1, 9, :],c='b')
plt.plot(range(10), Lat_str[step1, 10, :],c='r')
#plt.plot(range(10), Lat_str[step1, 11, :],c='g')
plt.plot(range(10), Lat_str[step1, 12, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 13, :],c='darkgoldenrod')

plt.text(5.2, 0.009, '0.2%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.subplot(2, 2, 2)
step1 = 24 #2%
for i in range(zzz):
    if i == 0:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40, label = plane_name[9])
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40, label = plane_name[10])
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40, label = plane_name[11])
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40, label = plane_name[12])
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[13])
    else:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40)
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40)
#lines between points
#plt.plot(range(10), Lat_str[step1, 9, :],c='b')
plt.plot(range(10), Lat_str[step1, 10, :],c='r')
#plt.plot(range(10), Lat_str[step1, 11, :],c='g')
plt.plot(range(10), Lat_str[step1, 12, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 13, :],c='darkgoldenrod')

plt.text(5.2, 0.009, '2%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.subplot(2, 2, 3)
step1 = 52 #10%
for i in range(zzz):
    if i == 0:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40, label = plane_name[9])
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40, label = plane_name[10])
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40, label = plane_name[11])
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40, label = plane_name[12])
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[13])
    else:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40)
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40)
#lines between points
#plt.plot(range(10), Lat_str[step1, 9, :],c='b')
plt.plot(range(10), Lat_str[step1, 10, :],c='r')
#plt.plot(range(10), Lat_str[step1, 11, :],c='g')
plt.plot(range(10), Lat_str[step1, 12, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 13, :],c='darkgoldenrod')

plt.text(5.2, 0.009, '10%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.subplot(2, 2, 4)
step1 = 83 #25%
for i in range(zzz):
    if i == 0:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40, label = plane_name[9])
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40, label = plane_name[10])
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40, label = plane_name[11])
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40, label = plane_name[12])
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40, label = plane_name[13])
    else:
#        plt.scatter(i, Lat_str[step1, 9, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[step1, 10, i], marker='v',c='r', s = 40)
#        plt.scatter(i, Lat_str[step1, 11, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[step1, 12, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[step1, 13, i], marker='*',c='darkgoldenrod', s = 40)
#lines between points
#plt.plot(range(10), Lat_str[step1, 9, :],c='b')
plt.plot(range(10), Lat_str[step1, 10, :],c='r')
#plt.plot(range(10), Lat_str[step1, 11, :],c='g')
plt.plot(range(10), Lat_str[step1, 12, :],c='darkslategray')
plt.plot(range(10), Lat_str[step1, 13, :],c='darkgoldenrod')

plt.text(5.2, 0.009, '25%', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.savefig('test2.jpg',dpi = 300, bbox_inches = 'tight')








