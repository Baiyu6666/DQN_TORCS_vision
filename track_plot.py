import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import spline
import scipy.signal as signal
import os
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'#
mpl.rcParams['axes.unicode_minus']= False
n_kernel = (32, 64, 64)
size_filter = (8, 4, 3)
stride = (4, 2, 1)
linear_n = (7744, 128, 32)

IMAGE_NUM = 3
N_ACTIONS = 6
N_STATE_IMG = IMAGE_NUM*64*64
N_STATE_LOW = 5
memory_MAX = 5000
LEARN_FREQUENCY = 500
SAVE_FREQUENCY = 1000 #save model and reward list
BATCH_SIZE = 32

track = np.loadtxt('data/Data_generate/track.csv',  delimiter=',')[2:]*7.5
dist = np.loadtxt('data/Data_generate/dist.csv',  delimiter=',')[2:]
sx = np.loadtxt('data/Data_generate/sx.csv',  delimiter=',')[2:]*80
plt.figure(figsize=(10, 6))
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('横向偏离/m', fontsize=15)
plt.xlim(0, 90)
plt.ylim(-6, 6)
plt.grid(ls='--')
plt.tick_params(labelsize=13)
step = np.arange(0,90,0.2)
plt.plot(step, track, linewidth=2.5)
plt.savefig('figure/data_track.jpg', bbox_inches='tight')


plt.figure(figsize=(10, 6))
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('行驶距离/m', fontsize=15)
plt.xlim(0, 90)
plt.ylim(0, 1700)
plt.grid(ls='--')
plt.tick_params(labelsize=13)
plt.plot(step, dist, linewidth=2.5)
plt.savefig('figure/data_dist.jpg', bbox_inches='tight')



plt.figure(figsize=(10, 6))
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('纵向速度/(km/h)', fontsize=15)
plt.xlim(0, 90)
plt.ylim(0, 80)
plt.grid(ls='--')
plt.tick_params(labelsize=13)
plt.plot(step, sx, linewidth=2.5)
plt.savefig('figure/data_sx.jpg', bbox_inches='tight')
plt.show()