from gym_torcs import TorcsEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import trange
import math


env = gym.make('MountainCar-v0')
env = env.unwrapped
env.render()


N_ACTIONS = env.action_space.n
N_STATE = env.observation_space.shape[0]
HIDDEN_DIM = 20
MOMERY_MAX = 2000
LEARN_FREQUENCY = 100
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.1
GAMMA = 0.9

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.out(x)
        return action_values


class DQN():
    def __init__(self):
        self.eval_net, self.tar_net = Net(N_STATE, HIDDEN_DIM, N_ACTIONS), Net(N_STATE, HIDDEN_DIM, N_ACTIONS)
        self.momery_counter = 0
        self.learn_counter = 0
        self.momery = np.zeros([MOMERY_MAX, N_STATE * 2 +2])
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        if np.random.uniform() < EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            eval_value = self.eval_net.forward(s).detach()
            action = torch.max(eval_value, -1)[1].data.numpy().tolist()
        return action

    def store_momery(self, s, a, r, s_):
        transaction = np.hstack((s, a, r, s_))
        index = self.momery_counter % MOMERY_MAX
        self.momery[index, :] = transaction
        self.momery_counter += 1

    def learn(self):
        if self.learn_counter % LEARN_FREQUENCY == 0:
            self.tar_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1
        sample = np.random.choice(MOMERY_MAX, BATCH_SIZE)
        b = self.momery[sample, :]
        b_s = torch.FloatTensor(b[:, :N_STATE])
        b_a = torch.LongTensor(b[:, N_STATE: N_STATE + 1])
        b_r = torch.FloatTensor(b[:, N_STATE + 1: N_STATE + 2])
        b_s_ = torch.FloatTensor(b[:, N_STATE + 2:])
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)
        q_next = self.tar_net.forward(b_s_).detach()
        q_tar = b_r + GAMMA * q_next.max(1)[0].reshape(-1, 1)
        loss = self.loss_fun(q_eval, q_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print("Start")
rlist = []
for i in trange(5000):
    s = env.reset()
    r_sum = 0
    #
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, _ = env.step(a)
        r_sum += r
        #r = s_[1] ** 2
        r = abs(s_[0] + 0.5)
        if done:
            r = 50
        dqn.store_momery(s, a, r, s_)
        if dqn.momery_counter > MOMERY_MAX:
            dqn.learn()
        if done:
            break
        s = s_
    print(r_sum)
    rlist.append(r_sum)

import matplotlib.pyplot as plt
plt.plot(rlist)


