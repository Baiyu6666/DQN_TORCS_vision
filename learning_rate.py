import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import scipy.signal as signal
import os

clr_list = ['red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c',
            'red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c',
            'red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c']

mpare'

distdir = []
rdir = []
radir = []
listdir = os.listdir('data/' + method + '/')
for dir in listdir:
    if '_dist_' in dir:
        distdir.append(dir)
    elif '_r_' in dir:
        rdir.append(dir)
    elif '_ra_' in dir:
        radir.append(dir)

for dir in [distdir, rdir, radir]:
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    for j in range(len(dir)):
        para = dir[j]
        episode = range(400)
        r_list = np.loadtxt('data/' + method + '/' + para, delimiter=',').reshape(len(episode), -1)

        mean = np.mean(r_list, 1)
        var = np.std(r_list, 1)
        #plt.plot(episode, signal.medfilt(mean, 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[j])
        plt.errorbar(episode, mean, yerr=var, fmt='-d', ms=0.8, elinewidth=0.8, capsize=2, ecolor=clr_list[j],
                     color=clr_list[j])

        # plt.plot(episode, signal.medfilt(r_list[1], 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[j])
    plt.legend(dir)
#plt.savefig('figure/compare_para/' + compare_para + '.jpg')
plt.show()