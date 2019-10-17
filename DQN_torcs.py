from multigym_torcs import TorcsEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
#import visdom
import pygame
import math
import os, time
from SumTree import Memory
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

manul = False
learn = True
data_generate = False
online_draw = False
method_list = [
                'nature_DQN',
               'DDQN',
               'pure_data_DQN',
               'reward_data_DQN',
                'pure_data_DDQN',
                'pure_data_DQN_0.98',
                'reward_data_DQN_0.98'
               ]


load_data = False
prepro_data = False
extra_reward = False
data_regenerate = False
prioritized = not True

N_ACTIONS = 5
N_STATE = 29
HIDDEN_DIM = 100
memory_MAX = 5000
LEARN_FREQUENCY = 100
BATCH_SIZE = 32
LR = 0.00075
GAMMA = 0.9
ACTION = (np.array([0.25, 0.0, 0]),
          np.array([-0.25, 0.0, 0]),
          # np.array([0.2, 0.0, 0]),
          # np.array([-0.2, 0.0, 0]),
          np.array([0.15, 0.0, 0]),
          np.array([-0.15, 0.0, 0]),
          # np.array([0.10, 0.0, 0]),
          # np.array([-0.10, 0.0, 0]),
          #np.array([0.05, 0.0, 0]),
          # np.array([-0.05, 0.0, 0]),
          np.array([0, 0.5, 0]),
          np.array([0, 0, 0]),)

N_RUN = 5
EXP_MAX = 140
EPI_MAX = 1000
N_DATA = 900
DATA_GAMMA = 0.95
TIME_PREPRO = 150



EXTRA_R = 20
EXTRA_R_GAMMA = 0.95

#pygame.init()
#screen = pygame.display.set_mode((600, 400))
#pygame.display.set_caption('pygame event')
env = TorcsEnv(port=3100, text_mode=not False, vision=False, throttle=True, gear_change=False)

# if online_draw:
#     vis = visdom.Visdom(env='torcs')


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
        self.memory_counter = 0
        self.learn_counter = 0
        if prioritized:
            self.memory = Memory(memory_MAX, 0)
        else:
            self.memory = np.zeros([memory_MAX, N_STATE * 2 +2])
        self.data_memory = N_DATA
        self.loss_fun = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)


    def choose_action(self, s):
        s = torch.FloatTensor(s)
        if np.random.uniform() < EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            eval_value = self.eval_net.forward(s).detach()
            if online_draw:
                vis.line(X=torch.Tensor([step]), Y=eval_value.data.reshape(1,N_ACTIONS), win='a_value',
                         update='append' if step > 4 else None, opts=dict(title='a_value',legend=['Left','Right','Accel','Hold','Brake']))
            action = torch.max(eval_value, -1)[1].data.numpy().tolist()
        return action

    def store_memory(self, s, a, r, s_, i):
        transaction = np.hstack((s, a, r, s_))
        if prioritized:
            self.memory.store(transaction)
        else:
            self.data_memory = int(N_DATA * DATA_GAMMA**i)
            transaction = np.hstack((s, a, r, s_))
            index = self.memory_counter % memory_MAX + \
                    self.data_memory * load_data * (self.memory_counter % memory_MAX == 0)
            self.memory[index, :] = transaction
        self.memory_counter += 1

    def save_memory(self):
        np.savetxt('data/' + method + '/memory.csv', self.memory, delimiter=',')

    def load_memory(self):
        self.memory = np.loadtxt('data/DDQN/memory.csv',  delimiter=',')
        self.memory_counter = N_DATA

    def learn(self, i, prepro=False):
        if self.learn_counter % LEARN_FREQUENCY == 0:
            self.tar_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1
        if prioritized:
            tree_idx, b, IS, b_isdemo = self.memory.sample(BATCH_SIZE, min(memory_MAX, self.memory_counter))
            rate = b_isdemo.sum() / BATCH_SIZE
            if self.learn_counter % 2 == 0 and online_draw and with_data:
                vis.line(X=torch.Tensor([self.learn_counter]), Y=rate.reshape(1, 1), win='Data use rate',
                         update='append', opts={'title': 'ratio'})

            IS = torch.Tensor(IS).to(device)
            b_isdemo = torch.Tensor(b_isdemo).to(device)

        else:
            sample = np.random.choice(min(self.memory_counter, memory_MAX) * (1 - prepro) + N_DATA * prepro, BATCH_SIZE)
            b = self.memory[sample, :]


        # if prepro:
        #     sample = np.random.choice(N_DATA, BATCH_SIZE)
        # else:
        #     sample = np.random.choice(memory_MAX, BATCH_SIZE)

        b_s = torch.FloatTensor(b[:, :N_STATE])
        b_a = torch.LongTensor(b[:, N_STATE: N_STATE + 1])
        b_r = torch.FloatTensor(b[:, N_STATE + 1: N_STATE + 2])
        b_s_ = torch.FloatTensor(b[:, N_STATE + 2:])
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)

        eval_value = self.eval_net.forward(b_s_).detach()
        a_ = torch.max(eval_value, -1)[1].data.reshape(-1, 1)
        tar_value = self.tar_net(b_s_).detach().gather(1, a_)
        q_tar = b_r + GAMMA * tar_value
        if prioritized:
            abs_errors = abs(q_eval.detach().cpu() - q_tar.detach().cpu())

            #print("learn")
            time_new = time.time()
            self.memory.batch_update(tree_idx, abs_errors)
            #print('update time:%.3f' % (time() - time_new))
            q_eval *= IS ** 0.5
            q_tar *= IS ** 0.5

        if extra_reward:
            b_r_e = [int(sample[k] <= self.data_memory) * EXTRA_R * EXTRA_R_GAMMA**i for k in range(BATCH_SIZE)]
            q_tar += torch.Tensor(b_r_e).reshape(-1, 1)

        loss = self.loss_fun(q_eval, q_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.learn_counter % 4 == 0 and online_draw:
            vis.line(X=torch.Tensor([self.learn_counter]), Y=loss.data.reshape(1,1), win='Loss', update='append' if self.learn_counter > 0 else None,  opts={'title': 'loss'})

    def save_model(self, run):
        if not os.path.exists('data/' + method):
            os.makedirs('data/' + method)
        torch.save(self.eval_net.state_dict(), 'data/' + method + '/%s_eval_net.pkl' % run)
        torch.save(self.tar_net.state_dict(), 'data/' + method + '/%s_tar_net.pkl' % run)

    def load_model(self, run):
        self.eval_net.load_state_dict(torch.load('data/' + method + '/%s_eval_net.pkl' % run))
        self.tar_net.load_state_dict(torch.load('data/' + method + '/%s_tar_net.pkl' % run))


print("Start")

for method in method_list[0]:
    r_list = np.zeros([N_RUN, EXP_MAX])
    for run in range(N_RUN):

        dqn = DQN()
        if data_generate:
            dqn.load_model(run)
        if load_data:
            dqn.load_memory()
            if prepro_data:
                for learn in trange(TIME_PREPRO):
                    dqn.learn(0, True)
                print('PreLearning end')

        for i in trange(EXP_MAX):

            s = env.reset()
            s = np.hstack(
                (s.angle, s.track, s.trackPos, s.speedX, s.speedY, s.speedZ, s.wheelSpinVel / 100.0, s.rpm))
            r_sum = 0

            if learn and not data_generate:
                EPSILON = 1 / 4 / math.log(i+2)
            else:
                EPSILON = 0.01

            for step in range(EPI_MAX):
                time_new = time.time()
                if not manul:
                    a = dqn.choose_action(s)
                else:
                    dqn.choose_action(s)
                # for event in pygame.event.get():
                #     if event.type == pygame.KEYDOWN:
                #         if event.key == pygame.K_DOWN:
                #             a = 4
                #         elif event.key == pygame.K_UP:
                #             a = 2


                #         elif event.key == pygame.K_LEFT:
                #             a = 0
                #         elif event.key == pygame.K_RIGHT:
                #             a = 1
                #         elif event.key == pygame.K_RSHIFT:
                #             manul = not manul
                #     else:
                #         a = 3
                a_drive = ACTION[a]

                s_, r, done, r1, r2, r3 = env.step(a_drive)
                r_sum += r
                if step>0:
                    time.sleep(0.05)
                #print('%.3f' %r)
                s_ = np.hstack(
                    (s_.angle, s_.track, s_.trackPos, s_.speedX, s_.speedY, s_.speedZ, s_.wheelSpinVel / 100.0, s_.rpm))
                dqn.store_memory(s, a, r, s_, i)

                if dqn.memory_counter >= N_DATA and data_generate:
                    dqn.save_memory()
                    print('Data generated')
                    os._exit(0)

                if dqn.memory_counter > memory_MAX and learn:
                    dqn.learn(i)

                if done:
                    r_list[run][i] = r_sum/step
                    print('step =', step, 'Run=', run+1, 'Method=', method, 'Memory=', dqn.memory_counter)
                    if online_draw:
                        vis.line(X=torch.Tensor([i]), Y=torch.Tensor([r_sum/step]), win='r_average',
                                 update='append' if dqn.learn_counter > 0 else None,  opts={'title': 'R_average'})
                    break

                if step % 2 == 0 and online_draw:
                    vis.line(X=torch.Tensor([step]), Y=torch.Tensor([r]), win='r',
                         update='append' if step > 0 else None,  opts={'title': 'r'})
                s = s_
                #print("%.3f" %(time.time()-time_new))

        dqn.save_model(run)
    if not data_regenerate and os.path.exists('data/' + method + '/r_list.csv'):
        r_list_o = np.loadtxt('data/' + method + '/r_list.csv', delimiter=',')
        r_list = np.vstack((r_list_o.T, r_list))
    np.savetxt('data/' + method + '/r_list.csv', r_list.T, delimiter=',')
env.end()
