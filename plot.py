import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import scipy.signal as signal
import os
method_list = [#'DQN_with_Data_priorited',
                'DDQN_priorited',
                #  #'nature_DQN',
                #'DDQN'
                # 'pure_data_DQN',
                # 'reward_data_DQN',
                # 'pure_data_DDQN',
                # 'pure_data_DQN_0.98',
                #  'reward_data_DQN_0.98'
               ]

clr_list = ['red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c',
            'red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c',
            'red', 'blue', 'g', 'k', 'gray', 'pink', 'm', 'c']

compare_para = 'EPSILON'
plt.figure(figsize=(12, 6))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
for i in range(len(method_list)):
    method = method_list[i]
    r_list = np.loadtxt('data/' + method + '/r_list.csv', delimiter=',')
    mean = np.mean(r_list, 1)

    episode = range(len(mean))
    #x_new = np.linspace(0, len(mean), 500)
    #y_new = spline(episode, mean, x_new)
    #plt.plot(x_new, y_new, marker='d', ms=0.8, linewidth=1.0, color=clr_list[i])
    plt.plot(episode, signal.medfilt(mean, 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[i])

plt.legend(method_list)


plt.figure(figsize=(12, 6))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
for i in range(len(method_list)):
    method = method_list[i]
    r_list = np.loadtxt('data/' + method + '/r_list.csv', delimiter=',')
    mean = np.mean(r_list, 1)
    var = np.std(r_list, 1)

    episode = range(len(mean))

    plt.errorbar(episode, mean, yerr=var, fmt='-d', elinewidth=0.8, capsize=2, ecolor=clr_list[i], color=clr_list[i])

plt.legend(method_list)

for i in range(len(method_list)):
    method = method_list[i]
    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(method)
    r_list = np.loadtxt('data/' + method + '/r_list.csv', delimiter=',').T
    episode = range(len(r_list[0, :]))
    for k in range(len(r_list[:, 0])):
        x_new = np.linspace(0, len(r_list[0, :]), 500)
        y_new = spline(episode, r_list[k, :], x_new)
        #plt.plot(x_new, y_new, marker='d', ms=0.8, linewidth=1.0, color=clr_list[k])
        plt.plot(episode, signal.medfilt(r_list[k, :], 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[k])

    plt.figure(figsize=(12, 6))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(method+' with para')
    episode = range(len(r_list[0]))
    listdir = os.listdir('data/' + method + '/' + compare_para)
    for j in range(len(listdir)):
        para = listdir[j]
        r_list = np.loadtxt('data/' + method + '/'+compare_para+'/' + para, delimiter=',').reshape(len(episode), -1)
        mean = np.mean(r_list, 1)
        plt.plot(episode, signal.medfilt(mean, 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[j])
        #plt.plot(episode, signal.medfilt(r_list[1], 9), marker='d', ms=0.8, linewidth=1.0, color=clr_list[j])
    plt.legend(listdir)
    plt.savefig('figure/compare_para/' + compare_para + '.jpg')
plt.show()


