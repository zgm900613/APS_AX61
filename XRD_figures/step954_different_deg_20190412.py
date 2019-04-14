# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:13:48 2019

@author: G Zhu
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

z = list(reversed([i for i in range(12)]))#设置colorbar
cmap = matplotlib.cm.get_cmap('brg')
normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
colors = [cmap(normalize(value)) for value in z]

def in_Excel(file_path):
    data_file = pd.read_table(file_path, skiprows = 13, sep = '\s+|,', header = None)
    dataF = data_file.ix[:, :]
    data_matrix = np.matrix(dataF)
    return(data_file, dataF, data_matrix)

file_names = os.listdir(r'E:\SynchrotronXray\APS20180423\XRDresults\AX61-10degs\Figures\step954_different_deg_20190412')
file_path1 = 'E:/SynchrotronXray/APS20180423/XRDresults/AX61-10degs/Figures/step954_different_deg_20190412/'

data_list = []
for i in tqdm(range(len(file_names))):
    file_path = file_path1 + file_names[i]
    data_file, data0F, data0 = in_Excel(file_path)
    data_list.append(data0)

plt.ylabel('Intensity', fontsize = 30, fontdict = font)
plt.xlabel('2$θ$ (°)', fontsize = 30, fontdict = font)
plt.rc('figure', figsize = (12,10))
#plt.rcParams['xtick.direction'] = 'in'
#plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(direction='in', length=6, width=2.5)
plt.rcParams['axes.linewidth'] = 2.5

for i in range(10):
    plt.plot(data_list[0][:, 0], data_list[i][:, 1] + i * 500, linewidth=2.5, c=colors[i+2])

plt.plot([6.17, 6.17], [0, 6000], '--', c='r')  #line
plt.plot([3.56, 3.56], [0, 6000], '--', c='r')  #line

for i in range(10):
    plt.text(7.99, 270+492*i, str(i*10)+'°', ha='right', fontsize = 22, fontname = 'Arial')   #hkil = 


plt.xticks(np.arange(3, 9, 1), fontsize = 26, fontname = 'Arial')
plt.yticks([])#np.arange(0, 7000, 1000), fontsize = 24, fontname = 'Arial')
plt.axis([3, 8, 0, 6000])

plt.savefig('step954_different_deg_20190412.jpg',dpi = 300, bbox_inches = 'tight')

plt.show()