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
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


font = {'family': 'Arial'}

z = list(reversed([i for i in range(100)]))#设置colorbar
cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
colors = [cmap(normalize(value)) for value in z]

cm = plt.cm.get_cmap('RdYlBu')

def in_Excel(file_path):
    data_file = pd.read_table(file_path, skiprows = 13, sep = '\s+|,', header = None)
    dataF = data_file.ix[:, :]
    data_matrix = np.matrix(dataF)
    return(data_file, dataF, data_matrix)

file_names = os.listdir(r'E:\SynchrotronXray\APS20180423\XRDresults\AX61-10degs\Figures\Angle202_212_different_strain_20190414_new')
file_path1 = 'E:/SynchrotronXray/APS20180423/XRDresults/AX61-10degs/Figures/Angle202_212_different_strain_20190414_new/'

data_list = []
for i in tqdm(range(len(file_names))):
    file_path = file_path1 + file_names[i]
    data_file, data0F, data0 = in_Excel(file_path)
    data_list.append(data0)
"""--Full------------------------------------------------------------------------"""
fig = plt.figure(figsize=(20,20))
ax = fig.gca(projection='3d')

for i in range(85): #裁掉高的一点
    for j in range(2500):
        if data_list[i][j, 1] > 16500:
            data_list[i][j, 1] = 16500

for i in tqdm(range(85)):
    ax.plot(np.squeeze(np.asarray(data_list[0][280:1700, 0])), # '''540:673的时候是（222）面 ''' 
            np.squeeze(np.asarray(data_list[i][280:1700, 1])), 
            i*0.3, zdir='y', c = colors[i+15], linewidth=2)

ax.set_xlim([3, 7.5])
ax.set_ylim([0, 26])
ax.set_zlim([0, 13000])

#ax.set_xlim([4, 4.5])#（222）放大图
#ax.set_ylim([0, 26])
#ax.set_zlim([300, 500])

ax.set_xlabel('2$θ$ (°)', fontsize = 30, fontdict = font, labelpad=15)
ax.set_ylabel('Engineering strain (%)', fontsize = 30, fontdict = font, labelpad=15)
ax.set_zlabel('Intensity', fontsize = 30, fontdict = font)

ax.set_xticklabels(np.arange(3, 8, 0.5), fontsize = 20)
ax.set_yticklabels(np.arange(0, 30, 5), fontsize = 20)
ax.set_zticklabels([])

ax.view_init(20, 290)
plt.savefig('Angle202_212_different_deg_20190412_full.jpg', dpi = 200, bbox_inches = 'tight')
plt.show()
"""--------------------------------------------------------------------------"""
"""----2D_plot----------------------------------------------------------------------"""
#z = list(reversed([i for i in range(100)]))#设置colorbar
#cmap = matplotlib.cm.get_cmap('viridis')
#normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
#colors = [cmap(normalize(value)) for value in z]
#
#
#plt.ylabel('Intensity', fontsize = 30, fontdict = font)
#plt.xlabel('2$θ$ (°)', fontsize = 30, fontdict = font)
#plt.rc('figure', figsize = (12,10))
##plt.rcParams['xtick.direction'] = 'in'
##plt.rcParams['ytick.direction'] = 'in'
#plt.tick_params(direction='in', length=6, width=2.5)
#plt.rcParams['axes.linewidth'] = 2.5
#
#for i in range(85):
#    plt.plot(data_list[0][:, 0], data_list[i][:, 1] + i * 50 +300, linewidth=2.5, c=colors[i+15])
#
#plt.plot([6.212, 6.212], [0, 6000], '--', c='r')  #line
#plt.plot([3.589, 3.589], [0, 6000], '--', c='r')  #line
#
##for i in range(10):
##    plt.text(7.99, 270+492*i, str(i*10)+'°', ha='right', fontsize = 22, fontname = 'Arial')   #hkil = 
#plt.text(7.99, 270, '0%', ha='right', fontsize = 22, fontname = 'Arial')
#plt.text(7.99, 4800, '25%', ha='right', fontsize = 22, fontname = 'Arial')
#
#plt.xticks(np.arange(3, 9, 1), fontsize = 26, fontname = 'Arial')
#plt.yticks([])#np.arange(0, 7000, 1000), fontsize = 24, fontname = 'Arial')
#plt.axis([3, 8, 0, 6000])
#
#plt.savefig('Angle202_212_different_deg_20190412_full(2D).jpg',dpi = 300, bbox_inches = 'tight')
#
#plt.show()



"""(222)面的放大图------------------------------------------------------------"""
#fig = plt.figure(figsize=(20,20))
#ax = fig.gca(projection='3d')
#for i in range(85): #裁掉高的一点
#    for j in range(2500):
#        if data_list[i][j, 1] > 16500:
#            data_list[i][j, 1] = 16500
#
#for i in tqdm(range(85)):
#    ax.plot(np.squeeze(np.asarray(data_list[0][540:673, 0])), # '''540:673的时候是（222）面 ''' 
#            np.squeeze(np.asarray(data_list[i][540:673, 1])), 
#            i*0.3, zdir='y', c = colors[i+15], linewidth=2)
#
#
#ax.set_xlim([4, 4.5])#（222）放大图
#ax.set_ylim([0, 26])
#ax.set_zlim([250, 550])
#
#ax.set_xlabel('2$θ$ (°)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_ylabel('Engineering strain (%)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_zlabel('Intensity', fontsize = 30, fontdict = font)
#
#ax.set_xticklabels((4.0, 4.1, 4.2, 4.3, 4.4, 4.5), fontsize = 20)
#ax.set_yticklabels(np.arange(0, 30, 5), fontsize = 20)
#ax.set_zticklabels([])
#
#ax.view_init(20, 290)
#plt.savefig('Angle202_212_different_deg_20190412_plane(222).jpg', dpi = 200, bbox_inches = 'tight')
#plt.show()
#

"""(10-11)面的放大图------------------------------------------------------------"""

#fig = plt.figure(figsize=(20,20))
#ax = fig.gca(projection='3d')
#for i in range(85): #裁掉高的一点
#    for j in range(2500):
#        if data_list[i][j, 1] > 16500:
#            data_list[i][j, 1] = 16500
#
#for i in tqdm(range(85)):
#    ax.plot(np.squeeze(np.asarray(data_list[0][468:568, 0])), # 
#            np.squeeze(np.asarray(data_list[i][468:568, 1])), 
#            i*0.3, zdir='y', c = colors[i+15], linewidth=2)
#
#
#ax.set_xlim([3.9, 4.2])#（222）放大图
#ax.set_ylim([0, 26])
#ax.set_zlim([250, 13000])
#
#ax.set_xlabel('2$θ$ (°)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_ylabel('Engineering strain (%)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_zlabel('Intensity', fontsize = 30, fontdict = font)
#
#ax.set_xticklabels((3.90, 3.95, 4.00, 4.05, 4.10, 4.15, 4.20), fontsize = 20)
#ax.set_yticklabels(np.arange(0, 30, 5), fontsize = 20)
#ax.set_zticklabels([])
#
#ax.view_init(50, 271)
#plt.savefig('Angle202_212_different_deg_20190412_plane(10-11)_angle(50_270).jpg', dpi = 200, bbox_inches = 'tight')
#plt.show()

"""(11-20)面的放大图------------------------------------------------------------"""
#fig = plt.figure(figsize=(20,20))
#ax = fig.gca(projection='3d')#

#for i in range(85): #裁掉高的一点
#    for j in range(2500):
#        if data_list[i][j, 1] > 16500:
#            data_list[i][j, 1] = 16500
#
#for i in tqdm(range(85)):
#    ax.plot(np.squeeze(np.asarray(data_list[0][1200:1268, 0])), # 271的时候【1200：:168】 
#            np.squeeze(np.asarray(data_list[i][1200:1268, 1])), 
#            i*0.3, zdir='y', c = colors[i+15], linewidth=2)
#
#
#ax.set_xlim([6.1, 6.3])#（222）放大图
#ax.set_ylim([0, 26])
#ax.set_zlim([250, 2500])
#
#ax.set_xlabel('2$θ$ (°)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_ylabel('Engineering strain (%)', fontsize = 30, fontdict = font, labelpad=15)
#ax.set_zlabel('Intensity', fontsize = 30, fontdict = font)
#
#ax.set_xticklabels((6.10, 6.125, 6.150, 6.175, 6.2, 6.225, 6.25, 6.275, 6.3), fontsize = 20)
#ax.set_yticklabels(np.arange(0, 30, 5), fontsize = 20)
#ax.set_zticklabels([])
#
#ax.view_init(20, 290)
#plt.savefig('Angle202_212_different_deg_20190412_plane(11-20).jpg', dpi = 200, bbox_inches = 'tight')
#plt.show()
"""-------------------------------------------------------------------------------"""

'''另一种方法画，按照z值涂色   '''
#ThreeD_data = np.zeros((2500*85, 3))
#for i in range(2500):
#    for j in range(85):
#        ThreeD_data[j*2500 + i, 0] = data_list[0][i, 1]
#        ThreeD_data[j*2500 + i, 1] = j
#        ThreeD_data[j*2500 + i, 2] = data_list[j][i, 1]
#    
#conc = np.vstack(ThreeD_data)
#df = pd.DataFrame(conc)
#
## And transform the old column name in something numeric
#df['X']=pd.Categorical(df['X'])
#df['X']=df['X'].cat.codes
# 
## Make the plot
#fig = plt.figure(figsize=(20,20))
#ax = fig.gca(projection='3d')
#ax.plot_trisurf(df['X'], df['Y'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#plt.show()
# 
## to Add a color bar which maps values to colors.
#surf=ax.plot_trisurf(df['X'], df['Y'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#fig.colorbar( surf, shrink=0.5, aspect=5)
#plt.show()
# 
## Rotate it
#ax.view_init(30, 45)
#plt.show()
# 
## Other palette
#ax.plot_trisurf(df['X'], df['Y'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
#plt.show()
'''另二种方法画，按照z值涂色   '''
#ThreeD_data2 = np.zeros((2500, 85))
#for i in range(2500):
#    for j in range(85):
#        ThreeD_data2[i, j] = data_list[j][i, 1]
#
#doc = open('E:/SynchrotronXray/APS20180423/XRDresults/AX61-10degs/Figures/3DData_test.txt', "w+")
#
#line_No, column_No = np.shape(ThreeD_data2)
#
#for i in range(line_No):
#    for j in range(column_No):
#        doc.write(str(ThreeD_data2[i, j]))
#        doc.write('\t')
#    doc.write('\n')
#doc.close()
