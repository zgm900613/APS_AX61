# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:32:54 2019

@author: G. Zhu
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay 
from matplotlib import cm

def cal_rot_Matrix(E1, E2, E3):
#    Euler = [0, 0, 0]
    Euler1 = E1 * math.pi / 180
    Euler2 = E2 * math.pi / 180
    Euler3 = E3 * math.pi / 180
    
    gT = np.zeros((3, 3))
    
    gT[0, 0] = math.cos(Euler1) * math.cos(Euler3) - math.sin(Euler1) * math.sin(Euler3) * math.cos(Euler2)
    gT[0, 1] = -math.cos(Euler1) * math.sin(Euler3) - math.sin(Euler1) * math.cos(Euler3) * math.cos(Euler2)
    gT[0, 2] = math.sin(Euler1) * math.sin(Euler2)
    gT[1, 0] = math.sin(Euler1) * math.cos(Euler3) + math.cos(Euler1) * math.sin(Euler3) * math.cos(Euler2)
    gT[1, 1] = - math.sin(Euler1) * math.sin(Euler3) + math.cos(Euler1) * math.cos(Euler3) * math.cos(Euler2)
    gT[1, 2] = - math.cos(Euler1) * math.sin(Euler2)
    gT[2, 0] = math.sin(Euler3) * math.sin(Euler2)
    gT[2, 1] = math.cos(Euler3) * math.sin(Euler2)
    gT[2, 2] = math.cos(Euler2)
    
    gM = gT.T
    
    return gM

def cal_position(initial_pos, gM):
    IP = np.zeros(3)
    for i in range(3):
        IP[i] = initial_pos[i]/(abs(initial_pos[0])+abs(initial_pos[1])+abs(initial_pos[2]))
    
    position = np.zeros(3)
    for i in range(3):
        position[i] = (gM[i, 0] * IP[0] + gM[i, 1] * IP[1] + gM[i, 2] * IP[2])*(abs(initial_pos[0])+abs(initial_pos[1])+abs(initial_pos[2]))
    return position

def draw_hcp(E1, E2, E3, pos_X, pos_Y, pos_Z, Color_No):
    
#    fig = plt.figure()
    color = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

#    ax = fig.gca(projection='3d')
    
    ax.text2D(0.1, 1, str(111) + ' plane', fontsize = 20, transform=ax.transAxes)

    c_over_a = 1.62
    a = 1
    
    initial_pos = np.zeros((12, 3))
    
    initial_pos[0] = [0, a, -c_over_a/2]
    initial_pos[1] = [math.sqrt(3)/2 * a, 0.5 * a, -c_over_a/2]
    initial_pos[2] = [math.sqrt(3)/2 * a, -0.5 * a, -c_over_a/2]
    initial_pos[3] = [0, -a, -c_over_a/2]
    initial_pos[4] = [-math.sqrt(3)/2 * a, -0.5 * a,  -c_over_a/2]
    initial_pos[5] = [-math.sqrt(3)/2 * a, 0.5 * a,  -c_over_a/2]
    
    initial_pos[6] = [0, a, c_over_a/2]
    initial_pos[7] = [math.sqrt(3)/2 * a, 0.5 * a, c_over_a/2]
    initial_pos[8] = [math.sqrt(3)/2 * a, -0.5 * a, c_over_a/2]
    initial_pos[9] = [0, -a, c_over_a/2]
    initial_pos[10] = [-math.sqrt(3)/2 * a, -0.5 * a, c_over_a/2]
    initial_pos[11] = [-math.sqrt(3)/2 * a, 0.5 * a, c_over_a/2]
    
    gM = cal_rot_Matrix(E1, E2, E3)
    
    atom_position = np.zeros((12, 3))
    for i in range(12):
        atom_position[i] = cal_position(initial_pos[i], gM)
        
        atom_position[i, 0] = atom_position[i, 0] + pos_X#位置的影响
        atom_position[i, 1] = atom_position[i, 1] + pos_Y
        atom_position[i, 2] = atom_position[i, 2] + pos_Z
        
        
    CH = Delaunay(initial_pos).convex_hull 
    x,y,z = initial_pos[:,0], initial_pos[:,1], initial_pos[:,2] 
    ax.plot_trisurf(x,y,z,triangles=CH ,shade = False, color = 'g', lw=0, alpha = 0.8) #cmap=cm.copper, 
#    ax.set_xticklabels([]) 
#    ax.set_yticklabels([]) 
#    ax.set_zticklabels([]) 


    for i in range(6):
        ax.plot([atom_position[i, 0], atom_position[i+6, 0]], [atom_position[i, 1], atom_position[i+6, 1]], [atom_position[i, 2], atom_position[i+6, 2]], '-', c=color[Color_No], linewidth = 1.5)
        if i != 5:
            ax.plot([atom_position[i, 0], atom_position[i+1, 0]], [atom_position[i, 1], atom_position[i+1, 1]], [atom_position[i, 2], atom_position[i+1, 2]], '-', c=color[Color_No], linewidth = 1.5)
            ax.plot([atom_position[i+6, 0], atom_position[i+7, 0]], [atom_position[i+6, 1], atom_position[i+7, 1]], [atom_position[i+6, 2], atom_position[i+7, 2]], '-', c=color[Color_No], linewidth = 1.5)
        else:
            ax.plot([atom_position[i, 0], atom_position[0, 0]], [atom_position[i, 1], atom_position[0, 1]], [atom_position[i, 2], atom_position[0, 2]], '-', c=color[Color_No], linewidth = 1.5)
            ax.plot([atom_position[i+6, 0], atom_position[6, 0]], [atom_position[i+6, 1], atom_position[6, 1]], [atom_position[i+6, 2], atom_position[+6, 2]], '-', c=color[Color_No], linewidth = 1.5)
    
    
#    ax.set_xlim(-2, 2)
#    ax.set_ylim(-2, 2)
#    ax.set_zlim(-2, 2)
    
    return 0

fig = plt.figure()
ax = fig.gca(projection='3d')


#for i in range(1):
#    draw_hcp(0,i,0,0,0,0,6)

draw_hcp(70,65,0,0,0,0,6)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

'''
https://stackoverflow.com/questions/4622057/plotting-3d-polygons-in-python-matplotlib

http://matplotlib.1069221.n5.nabble.com/matplotlib-3D-interpolated-shading-td7989.html
'''

















#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#ax.text2D(0.1, 1, str(111) + ' plane', fontsize = 20, transform=ax.transAxes)
#
#
##ax.plot([0,1], [0,1], [0,1], '-', c='r', linewidth = 2)
#
#c_over_a = 1.62
#a = 1
#
#atom_position = np.zeros((12, 3))
#
#atom_position[0] = [0, a, 0]
#atom_position[1] = [math.sqrt(3)/2 * a, 0.5 * a, 0]
#atom_position[2] = [math.sqrt(3)/2 * a, -0.5 * a, 0]
#atom_position[3] = [0, -a, 0]
#atom_position[4] = [-math.sqrt(3)/2 * a, -0.5 * a,  0]
#atom_position[5] = [-math.sqrt(3)/2 * a, 0.5 * a,  0]
#
#atom_position[6] = [0, a, c_over_a]
#atom_position[7] = [math.sqrt(3)/2 * a, 0.5 * a, c_over_a]
#atom_position[8] = [math.sqrt(3)/2 * a, -0.5 * a, c_over_a]
#atom_position[9] = [0, -a, c_over_a]
#atom_position[10] = [-math.sqrt(3)/2 * a, -0.5 * a, c_over_a]
#atom_position[11] = [-math.sqrt(3)/2 * a, 0.5 * a, c_over_a]
#
#
#
#
#
#
#for i in range(6):
#    ax.plot([atom_position[i, 0], atom_position[i+6, 0]], [atom_position[i, 1], atom_position[i+6, 1]], [atom_position[i, 2], atom_position[i+6, 2]], '-', c='b', linewidth = 1.5)
#    if i != 5:
#        ax.plot([atom_position[i, 0], atom_position[i+1, 0]], [atom_position[i, 1], atom_position[i+1, 1]], [atom_position[i, 2], atom_position[i+1, 2]], '-', c='b', linewidth = 1.5)
#        ax.plot([atom_position[i+6, 0], atom_position[i+7, 0]], [atom_position[i+6, 1], atom_position[i+7, 1]], [atom_position[i+6, 2], atom_position[i+7, 2]], '-', c='b', linewidth = 1.5)
#    else:
#        ax.plot([atom_position[i, 0], atom_position[0, 0]], [atom_position[i, 1], atom_position[0, 1]], [atom_position[i, 2], atom_position[0, 2]], '-', c='b', linewidth = 1.5)
#        ax.plot([atom_position[i+6, 0], atom_position[6, 0]], [atom_position[i+6, 1], atom_position[6, 1]], [atom_position[i+6, 2], atom_position[+6, 2]], '-', c='b', linewidth = 1.5)





#ax.plot([math.sqrt(3)/2 * a, math.sqrt(3)/2 * a], [0.5 * a, -0.5 * a], [0, 0], '-', c='b', linewidth = 1.5)
#ax.plot([math.sqrt(3)/2 * a, 0], [-0.5 * a, -a], [0, 0], '-', c='b', linewidth = 1.5)
#ax.plot([0, -math.sqrt(3)/2 * a], [-a, -0.5 * a], [0, 0], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, -math.sqrt(3)/2 * a], [-0.5 * a, 0.5 * a], [0, 0], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, 0], [0.5 * a, a], [0, 0], '-', c='b', linewidth = 1.5)
#
#ax.plot([0, 0], [a, a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([math.sqrt(3)/2 * a, math.sqrt(3)/2 * a], [0.5 * a, 0.5 * a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([math.sqrt(3)/2 * a, math.sqrt(3)/2 * a], [-0.5 * a, -0.5 * a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([0, 0], [-a, -a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, -math.sqrt(3)/2 * a], [-0.5 * a, -0.5 * a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, -math.sqrt(3)/2 * a], [0.5 * a, 0.5 * a], [0, c_over_a], '-', c='b', linewidth = 1.5)
#
#ax.plot([0, math.sqrt(3)/2 * a], [a, 0.5 * a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([math.sqrt(3)/2 * a, math.sqrt(3)/2 * a], [0.5 * a, -0.5 * a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([math.sqrt(3)/2 * a, 0], [-0.5 * a, -a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([0, -math.sqrt(3)/2 * a], [-a, -0.5 * a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, -math.sqrt(3)/2 * a], [-0.5 * a, 0.5 * a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)
#ax.plot([-math.sqrt(3)/2 * a, 0], [0.5 * a, a], [c_over_a, c_over_a], '-', c='b', linewidth = 1.5)

