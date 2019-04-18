# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:43:57 2019

@author: G. Zhu
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from scipy.interpolate import griddata

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
Area = np.zeros((86, 14, 10))
xxx, yyy, zzz = np.shape(Lat_str)

plane_name = ('(10-10)', '(0002)', '(10-11)', '(10-12)', '(11-20)', 
              '(10-13)', '(20-20)', '(11-22)', '(20-21)', 
              '(220)', '(222)', '(422)', '(511)', '(440)')
strain_name = ('0.2%', '1.0%', '2.0%', '5.0%', '10.0%', '25.0%')

for i in tqdm(range(xxx)):
    for j in range(yyy-5):
        for k in range(zzz):
            Lat_str[i, j, k] = data_list[k][i+5, 10*j+17]
    for j in range(5):
        for k in range(zzz):
            Lat_str[i, j+9, k] = data_list[k][i+5, 10*j+110]

for i in tqdm(range(xxx)):
    for j in range(yyy-5):
        for k in range(zzz):
            Area[i, j, k] = data_list[k][i+5, 10*j+12]
    for j in range(5):
        for k in range(zzz):
            Area[i, j+9, k] = data_list[k][i+5, 10*j+105]

#for i in range(xxx):
#    for j in range(zzz):
#        if Lat_str[i, 13, j] > 0.014 or Lat_str[i, 13, j] < -0.003:
#            Lat_str[i, 13, j] = 0.00

#X = np.arange(0, 86, 1)
X = data_list[0][5:90, 8]
Y = np.arange(0, 10, 1)

Plane_x = 1
X, Y = np.meshgrid(Y, X)
Z = Lat_str[:85, Plane_x, :]

Y = np.array(Y)
X = np.array(X)
Z = np.array(Z)


fig = plt.figure(figsize=(18,15))
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-0.005, 0.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.text2D(0.1, 1, str(plane_name[Plane_x]) + ' plane', fontsize = 20, transform=ax.transAxes) 
#ax.text(1, 1, 1, "red", color='red')

ax.set_xlabel('Integration angle to the tensile direction (deg.)', fontsize = 20, fontdict = font, labelpad=15)
ax.set_ylabel('Engineering strain (%)', fontsize = 20, fontdict = font, labelpad=15)
ax.set_zlabel('Lattice strain', fontsize = 20, fontdict = font, labelpad=25)


ax.set_xlim([0, 9])
ax.set_ylim([0, 30])
ax.set_zlim([-0.004, 0.014])

ax.set_xticklabels((0, 10, 20, 30, 40, 50, 60, 70, 80, 90), fontsize = 20)
ax.set_yticklabels((0, 5, 10, 15, 20, 25), fontsize = 20)
ax.set_zticklabels((-0.004, -0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014), fontsize = 20)  #(-0.004, -0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(20, 290)
#plt.savefig('3D_surface_Lattice_strain' + str(plane_name[Plane_x]) + '20190417.jpg', dpi = 200, bbox_inches = 'tight')
plt.show()



#https://stackoverflow.com/questions/35157650/smooth-surface-plot-with-pyplot
#
#xnew, ynew = np.mgrid[-1:1:80j, -1:1:80j]
#tck = interpolate.bisplrep(X, Y, Z, s=0)
#znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
#
#fig = plt.figure(figsize=(12,12))
#ax = fig.gca(projection='3d')
#ax.plot_surface(xnew, ynew, znew, cmap='summer', rstride=1, cstride=1, alpha=None, antialiased=True)
#plt.show()






