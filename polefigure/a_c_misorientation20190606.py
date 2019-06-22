# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:23:24 2019

@author: GM Zhu
"""

import pandas as pd
import numpy as np
import math
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt


def intxt(file_path):   #define a function to import log file
    x = pd.read_table(file_path, skiprows = 19, sep = '\s+|,', header = None)
    xM = x.ix[:, 0 : 5]
    xMatrix = np.matrix(xM)
    return(xMatrix)


def in_Excel(file_path):
    data_file = pd.read_excel(file_path, skip = 0)
    dataF = data_file.ix[:, :]
    data_matrix = np.matrix(dataF)
    return(data_file, dataF, data_matrix)

def cal_rot_matrix(E1, E2, E3):
    '''#因为原来就是用的逆矩阵，所以要再反回来
    calculate rotation matrix of a given Euler angles E1, E2, E3
    '''
    i = 0
    gT_by_one = np.zeros((3, 3))
    Euler_radian = np.zeros((1, 3))
    Euler_radian[0, 0] = E1 * math.pi / 180
    Euler_radian[0, 1] = E2 * math.pi / 180
    Euler_radian[0, 2] = E3 * math.pi / 180

    gT_by_one[0, 0] = math.cos(Euler_radian[i, 0]) * math.cos(Euler_radian[i, 2]) - math.sin(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 2]) * math.cos(Euler_radian[i, 1])
    gT_by_one[0, 1] = -math.cos(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 2]) - math.sin(Euler_radian[i, 0]) * math.cos(Euler_radian[i, 2]) * math.cos(Euler_radian[i, 1])
    gT_by_one[0, 2] = math.sin(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 1])
    gT_by_one[1, 0] = math.sin(Euler_radian[i, 0]) * math.cos(Euler_radian[i, 2]) + math.cos(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 2]) * math.cos(Euler_radian[i, 1])
    gT_by_one[1, 1] = - math.sin(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 2]) + math.cos(Euler_radian[i, 0]) * math.cos(Euler_radian[i, 2]) * math.cos(Euler_radian[i, 1])
    gT_by_one[1, 2] = - math.cos(Euler_radian[i, 0]) * math.sin(Euler_radian[i, 1])
    gT_by_one[2, 0] = math.sin(Euler_radian[i, 2]) * math.sin(Euler_radian[i, 1])
    gT_by_one[2, 1] = math.cos(Euler_radian[i, 2]) * math.sin(Euler_radian[i, 1])
    gT_by_one[2, 2] = math.cos(Euler_radian[i, 1])
    return np.matrix(gT_by_one)


'''
polefigure
'''
Plane_No = 3  #1(100)2(110)3(111)


data_file, data0F, data0 = in_Excel(r'C:\Users\GM Zhu\Desktop\Al2Ca_EBSD\Mg_Al2Ca_Mis.xlsx')

Euler_Mg = np.zeros((len(data0), 3))
Euler_Al2Ca = np.zeros((len(data0), 3))


for i in range(len(data0)):
    for j in range(3):
        Euler_Mg[i, j] = data0[i, j]
        Euler_Al2Ca[i, j] = data0[i, j+4]

RotM_Mg = []
RotM_Al2Ca = []
for i in range(len(data0)):
    rotxx_Mg = cal_rot_matrix(Euler_Mg[i, 0], Euler_Mg[i, 1], Euler_Mg[i, 2])
    rotxx_Al2Ca = cal_rot_matrix(Euler_Al2Ca[i, 0], Euler_Al2Ca[i, 1], Euler_Al2Ca[i, 2])
    RotM_Mg.append(rotxx_Mg)
    RotM_Al2Ca.append(rotxx_Al2Ca)


Mg_planeN0001 = np.array([[0, 0, 1]])

if Plane_No == 1:
    Al2Ca_planeN001 = np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0],
                              [0, 0, -1],
                              [0, -1, 0],
                              [-1, 0, 0]])
if Plane_No == 2:
    Al2Ca_planeN001 = np.array([[1, 1, 0],
                              [1, -1, 0],
                              [-1, 1, 0],
                              [-1, -1, 0],
                              [1, 0, 1],
                              [1, 0, -1],
                              [-1, 0, 1],
                              [-1, 0, -1],
                              [0, 1, 1],
                              [0, 1, -1],
                              [0, -1, 1],
                              [0, -1, -1]])/math.sqrt(2)

if Plane_No == 3:
    Al2Ca_planeN001 = np.array([[1, 1, 1],
                              [1, 1, -1],
                              [1, -1, 1],
                              [1, -1, -1],
                              [-1, 1, 1],
                              [-1, 1, -1],
                              [-1, -1, 1],
                              [-1, -1, -1]])/math.sqrt(3)

    
        
New_Al2Ca_PlaneN = np.zeros((len(data0), len(Al2Ca_planeN001)*3))#123 are for 001, 456 are for 010, 789 are for 100

for i in range(len(data0)):
    for j in range(len(Al2Ca_planeN001)):
        New_Al2Ca_PlaneN[i, 0+3*j], New_Al2Ca_PlaneN[i, 1+3*j], New_Al2Ca_PlaneN[i, 2+3*j] = np.array(np.array(Al2Ca_planeN001[j] * RotM_Al2Ca[i])[0] * RotM_Mg[i].T)[0]

x = np.zeros((int(len(data0)*len(Al2Ca_planeN001)/2), 1))
y = np.zeros((int(len(data0)*len(Al2Ca_planeN001)/2), 1))

k=0
for i in range(len(data0)):
    for j in range(len(Al2Ca_planeN001)):
        if New_Al2Ca_PlaneN[i, 2+3*j] > 0:
            x[k, 0] = New_Al2Ca_PlaneN[i, 0+3*j]
            y[k, 0] = New_Al2Ca_PlaneN[i, 1+3*j]
            k+=1

fig = plt.figure()
ax = fig.add_subplot(111)
radius = 1
cir1 = Circle((0, 0), radius, clip_on=False, linewidth=2.5, edgecolor='green', facecolor=(0, 0, 0, 0))
ax.add_patch(cir1)
plt.axis('scaled')
plt.axis('equal') 
ax.scatter(x, y, marker='o', c='', edgecolors='b', linewidth = 2.5, s = 70, alpha = 0.8)#linewidth = 0.5, s = 30,

ax.scatter(0, 0, marker = 'o', c='', edgecolors='g', linewidth = 3, s = 10, alpha = 0.8)
cir2 = Circle((0, 0), 0.5, clip_on=False, linewidth=0.5, edgecolor='g', facecolor=(0, 0, 0, 0))
ax.add_patch(cir2)



ax.plot([0, 0], [1, -1], 'g--', linewidth=0.5)
ax.plot([-1, 1], [0, 0], 'g--', linewidth=0.5)



plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.xlim((-1.1, 1.1))
plt.ylim((-1.1, 1.1))
plt.rc('figure', figsize=(10,10))
if Plane_No == 1:
    plt.title('Polefigure (100)', fontsize = 18)
    plt.savefig('polefigure(100).jpg',dpi = 300, bbox_inches = 'tight')
if Plane_No == 2:
    plt.title('Polefigure (110)', fontsize = 18)
    plt.savefig('polefigure(110).jpg',dpi = 300, bbox_inches = 'tight')
if Plane_No == 3:
    plt.title('Polefigure (111)', fontsize = 18)
    plt.savefig('polefigure(111).jpg',dpi = 300, bbox_inches = 'tight')


plt.show()


    
    
    
    
 
'''
cal plane angle and direction angle
'''
#
#data_file, data0F, data0 = in_Excel(r'C:\Users\GM Zhu\Desktop\Al2Ca_EBSD\Mg_Al2Ca_Mis.xlsx')
#
#Mg_dir = np.array([[0, 1, 0],
#          [math.sqrt(3)/2, 0.5, 0],
#          [math.sqrt(3)/2, -0.5, 0]])
#
#    
##Mg_dir = np.array([[1, 0, 0],
##          [0.5, math.sqrt(3)/2, 0],
##          [-0.5, math.sqrt(3)/2, 0]])
#
#
#
#Al2Ca_dir = np.array([[1, 0, 1],
#           [1, 0, -1],
#           [1, 1, 0],
#           [1, -1, 0],
#           [0, 1, 1],
#           [0, 1, -1]])
#
#k = 2
#
#Angle = 180
#for i in range(3):
#    for j in range(6):
#        New_Mg_dir = np.array(Mg_dir[i]*cal_rot_matrix(data0[k, 1], data0[k, 2], data0[k, 3]))
#        New_Al2Ca_dir = np.array(Al2Ca_dir[j]*cal_rot_matrix(data0[k, 5], data0[k, 6], data0[k, 7]))
#        Lx = np.sqrt(New_Mg_dir[0].dot(New_Mg_dir[0]))
#        Ly = np.sqrt(New_Al2Ca_dir[0].dot(New_Al2Ca_dir[0]))
#        Test_angle = math.acos(New_Mg_dir[0].dot(New_Al2Ca_dir[0])/(Lx*Ly))*180 /math.pi
#        if Test_angle > 90:
#            Test_angle = 180-Test_angle
#        if Test_angle < Angle:
#            Angle = Test_angle
#    
#print(Angle)
#    
#Mg_plane = np.array([[0, 0, 1]])
#
#Al2Ca_plane = np.array([[1, 1, 1],
#                        [1, 1, -1],
#                        [1, -1, 1],
#                        [1, -1, -1]])
#    
#Angle = 180
#for i in range(1):
#    for j in range(4):
#        New_Mg_plane = np.array(Mg_plane[i]*cal_rot_matrix(data0[k, 1], data0[k, 2], data0[k, 3]))
#        New_Al2Ca_plane = np.array(Al2Ca_plane[j]*cal_rot_matrix(data0[k, 5], data0[k, 6], data0[k, 7]))
#        Lx = np.sqrt(New_Mg_plane[0].dot(New_Mg_plane[0]))
#        Ly = np.sqrt(New_Al2Ca_plane[0].dot(New_Al2Ca_plane[0]))
#        Test_angle = math.acos(New_Mg_plane[0].dot(New_Al2Ca_plane[0])/(Lx*Ly))*180 /math.pi
#        if Test_angle > 90:
#            Test_angle = 180-Test_angle
#        if Test_angle < Angle:
#            Angle = Test_angle
#
#print(Angle)
'''
---------------------------------------------------------------------------------------------
'''
       
    
    
    
    

