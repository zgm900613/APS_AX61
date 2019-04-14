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
import matplotlib
from matplotlib.lines import Line2D


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

strain_name = ('0.2%', '1.0%', '2.0%', '5.0%', '10.0%', '25.0%')

for i in tqdm(range(xxx)):
    for j in range(yyy-5):
        for k in range(zzz):
            Lat_str[i, j, k] = data_list[k][i+5, 10*j+17]
    for j in range(5):
        for k in range(zzz):
            Lat_str[i, j+9, k] = data_list[k][i+5, 10*j+110]

plt.figure(figsize = (15, 25))

'''------------------------------------------------------------------------------'''

plt.subplot(3, 2, 1)
plane1 = 0
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(10-10)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 2)
plane1 = 2
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(10-11)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 3)
plane1 = 3
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(10-12)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 4)
plane1 = 4
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(11-20)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 5)
plane1 = 5
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(10-13)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 6)
plane1 = 7
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(11-22)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.savefig('test_plane.jpg',dpi = 300, bbox_inches = 'tight')


















plt.figure(figsize = (15, 25))

'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 1)
plane1 = 10
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(222)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
#plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 2)
plane1 = 12
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(511)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

#plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 3)
plane1 = 13
for i in range(zzz):
    if i == 0:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40, label = strain_name[0])
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40, label = strain_name[1])
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40, label = strain_name[2])
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40, label = strain_name[3])
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40, label = strain_name[4])
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40, label = strain_name[5])
    else:
        plt.scatter(i, Lat_str[12, plane1, i], marker='o',c='b', s = 40)
        plt.scatter(i, Lat_str[19, plane1, i], marker='v',c='r', s = 40)
        plt.scatter(i, Lat_str[24, plane1, i], marker='^',c='g', s = 40)
        plt.scatter(i, Lat_str[37, plane1, i], marker='p',c='darkslategray', s = 40)
        plt.scatter(i, Lat_str[52, plane1, i], marker='*',c='darkgoldenrod', s = 40)
        plt.scatter(i, Lat_str[83, plane1, i], marker='s',c='m', s = 40)
#lines between points
plt.plot(range(10), Lat_str[12, plane1, :],c='b')
plt.plot(range(10), Lat_str[19, plane1, :],c='r')
plt.plot(range(10), Lat_str[24, plane1, :],c='g')
plt.plot(range(10), Lat_str[37, plane1, :],c='darkslategray')
plt.plot(range(10), Lat_str[52, plane1, :],c='darkgoldenrod')
plt.plot(range(10), Lat_str[83, plane1, :],c='m')

plt.text(5.3, 0.009, '(440)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.0125, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.01])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})
'''------------------------------------------------------------------------------'''

plt.savefig('test_plane2.jpg',dpi = 300, bbox_inches = 'tight')












plt.figure(figsize = (15, 25))

'''------------------------------------------------------------------------------'''
plt.subplot(3, 2, 1)
plane1 = 10

z = list(reversed([i for i in range(86)]))
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
colors = [cmap(normalize(value)) for value in z]

custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]#colorbar

for i in range(zzz):
    for j in range(86):
        if i == 0:
            plt.scatter(i, Lat_str[j, plane1, i], marker='o',c=colors[j], s = 40)#, label = strain_name[0])
        else:
            plt.scatter(i, Lat_str[j, plane1, i], marker='o',c=colors[j], s = 40)
#lines between points
for j in range(86):
    plt.plot(range(10), Lat_str[j, plane1, :],c=colors[j])

plt.text(5.3, 0.0115, '(222)', ha = 'right', fontsize = 24, fontname = 'Arial')

plt.xticks(ind, ('0', '10', '20', '30', '40', '50', '60', '70', '80', '90'), fontsize = 17, fontname = 'Arial')
plt.yticks(np.arange(-0.0025, 0.015, 0.0025), fontsize = 17, fontname = 'Arial')
plt.axis([0, 9, -0.0025, 0.0125])

plt.ylabel('Lattice strain', fontsize = 20, fontdict = font)
plt.xlabel('Integration angle to tensile direction (deg.)', fontsize = 20, fontdict = font)
plt.tick_params(direction = 'in', length = 5, width = 1.5)
plt.rcParams['axes.linewidth'] = 2
plt.legend(prop={'family':'Arial', 'size':15})

plt.legend(custom_lines, ['25%', '13%', '0%'], fontsize = 17)#colorbar
'''------------------------------------------------------------------------------'''
plt.savefig('test_plane3.jpg',dpi = 300, bbox_inches = 'tight')



