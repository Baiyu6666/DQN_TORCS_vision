

from multigym_torcs import TorcsEnv
#from gym_torcs import TorcsEnv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
#import visdom
import math
import os
from SumTree import Memory
from CNN import CNN
from Utils import *


import argparse
from time import time, sleep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION = (
            np.array([0.25, 0.0, 0]),
            np.array([0.20, 0.0, 0]),
            np.array([0.15, 0.0, 0]),
            np.array([0.10, 0.0, 0]),
            np.array([0.05, 0.0, 0]),
            np.array([0.02, 0.0, 0]),
            np.array([0.01, 0.0, 0]),
            np.array([0.005, 0.0, 0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.005, 0.0, 0]),
            np.array([-0.01, 0.0, 0]),
            np.array([-0.02, 0.0, 0]),
            np.array([-0.05, 0.0, 0]),
            np.array([-0.10, 0.0, 0]),
            np.array([-0.15, 0.0, 0]),
            np.array([-0.20, 0.0, 0]),
            np.array([-0.25, 0.0, 0])
            )

# method_set = (('DDQN', True, False, False, False, False),
#               ('DDQN_priorited', True, False, False, True, False),
#               ('DQN_with_Data', T
#
#               rue, False, True, False, False),
#               ('DQN_with_Data_priorited_sup', True, False, True, True, True),
#               ('Data_generate', False, True, False, False, False),
#               ('DQN_with_Data_priorited', True, False, True, True, False)
#               )
# method, learn, data_generate, with_data, prioritized, supervised = method_set[4]
method = 'PDD'
para = 'final'
retrain = True
learn = True
prioritized = not True

#Function
data_generate = False
a_replay = False
a_record = False

record = False
evaluation = True
manul = False
online_draw = False


IMAGE_NUM = 1
N_ACTIONS = len(ACTION)
N_STATE_IMG = IMAGE_NUM*64*64
N_STATE_LOW = 5

#hyper-para
memory_MAX = 10000
memory_learn = 1000
LEARN_FREQUENCY = 500
SAVE_FREQUENCY = 2000
BATCH_SIZE = 32
LR = 0.0003
LR_delta = 1.7 * 10**(-7)
GAMMA = 0.9
MAX_EPISODE = 700
MAX_STEP = 1000
EPSILON = 0.1

#Data
with_data = not True
supervised = not True
N_DATA = 2500 #利用数据的比例大约是30%~50%
TIME_PREPRO = 200 #1000~5000左右
lamda = 0.2
margin = 0.1
memory_load = 'CNN0.15.csv'
memory_save = 'test.csv'

#Parameters for CNN
n_kernel = (32, 64, 64)
size_filter = (8, 4, 3)
stride = (4, 2, 1)
linear_n = (7744, 128, 32)


def args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--p', type=int, default=3100)
    args = parser.parse_args()

    return args
class DQN():
    def __init__(self):
        self.eval_net = CNN(n_kernel, size_filter, stride, linear_n, IMAGE_NUM, N_STATE_LOW, N_ACTIONS).to(device)
        self.tar_net = CNN(n_kernel, size_filter, stride, linear_n, IMAGE_NUM, N_STATE_LOW, N_ACTIONS).to(device)
        #self.console = Console()

        self.memory_counter = 0
        self.learn_counter = 0
        self.actions = [0 for _ in range(N_ACTIONS)]
        self.loss_fun = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        if prioritized:
            self.memory = Memory(memory_MAX, N_DATA*with_data)
        else:
            self.memory = np.zeros((memory_MAX, N_STATE_IMG * 2 + N_STATE_LOW * 2 + 2))
        if not os.path.exists('data/' + method):
            os.makedirs('data/' + method)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = LR - LR_delta * epoch
        lr *= 2
        if self.learn_counter % 2 == 0 and online_draw:
            vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([lr]), win='lr',
                     update='append' ,
                     opts=dict(title='lr'))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def choose_action(self, s_image, s_low):
        s_image = torch.Tensor(s_image).to(device)
        s_low = torch.Tensor(s_low).to(device)
        eval_value = self.eval_net.forward(s_image, s_low).detach()

        if np.random.uniform() < EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            action = int(torch.max(eval_value, -1)[1])

        if self.learn_counter % 2 == 0 and online_draw:
            vis.line(X=torch.Tensor([step]), Y=eval_value.data.reshape(1, N_ACTIONS), win='a_value',
                     update='append' if step > 4 else None, opts=dict(title='a_value', legend=['Left25', 'Right25', 'Left10', 'Right10', 'Accel', 'Hold']))
            vis.bar(X=eval_value.data.reshape(1, N_ACTIONS), win='a_value_h',
                      opts=dict(title='a_value', columnnames=['Left25', 'Right25', 'Left10', 'Right10', 'Accel', 'Hold']))
        if online_draw:
            if action == 0:
                vis.text('<---', win='a')
            elif action == 1:
                vis.text('--->', win='a')
            elif action == 2:
                vis.text('<-', win='a')
            elif action == 3:
                vis.text('->', win='a')
            elif action == 4:
                vis.text('|', win='a')
            elif action == 5:
                vis.text('o', win='a')
        if record:
            print(eval_value.data.reshape(1, N_ACTIONS))
        return action

    def action_counter(self, a):
        self.actions[a] += 1
        if self.learn_counter % 2 == 0 and online_draw:
            vis.bar(self.actions, win='a_counter', opts=dict(title='a_counter'))

    def store_memory(self, s_img, s_low, a, r, s_img_, s_low_):
        transaction = np.hstack((s_img.reshape(-1), s_low, a, r, s_img_.reshape(-1),
                                 s_low_))
        if prioritized:
            self.memory.store(transaction)
        else:
            index = self.memory_counter % memory_MAX + \
                    N_DATA * with_data * (self.memory_counter % memory_MAX == 0)
            self.memory[index] = transaction
        self.memory_counter += 1

    def learn(self, prepro=False):
        rate = 0
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
            sample = np.random.choice(min(self.memory_counter, memory_MAX)*(1-prepro)+N_DATA*prepro, BATCH_SIZE)
            b = self.memory[sample, :]

        b_s_img = torch.FloatTensor(b[:, :N_STATE_IMG]).reshape(BATCH_SIZE, IMAGE_NUM, 64, 64).to(device)
        b_s_low = torch.FloatTensor(b[:, N_STATE_IMG: N_STATE_IMG + N_STATE_LOW]).to(device)
        b_a = torch.LongTensor(b[:, N_STATE_IMG+N_STATE_LOW: N_STATE_IMG+N_STATE_LOW + 1]).to(device)
        b_r = torch.FloatTensor(b[:, N_STATE_IMG+N_STATE_LOW + 1: N_STATE_IMG+N_STATE_LOW + 2]).to(device)
        b_s_img_ = torch.FloatTensor(b[:, N_STATE_IMG+N_STATE_LOW+2:N_STATE_IMG*2+N_STATE_LOW+2]).reshape(BATCH_SIZE, IMAGE_NUM, 64, 64).to(device)
        b_s_low_ = torch.FloatTensor(b[:, N_STATE_IMG*2+N_STATE_LOW+2:N_STATE_IMG*2+N_STATE_LOW*2+2]).to(device)

        q_evals = self.eval_net.forward(b_s_img, b_s_low)
        q_eval = q_evals.gather(1, b_a)
        eval_value = self.eval_net.forward(b_s_img_, b_s_low_).detach()
        a_ = torch.max(eval_value, -1)[1].data.reshape(-1, 1)
        tar_value = self.tar_net(b_s_img_, b_s_low_).detach().gather(1, a_)
        q_tar = b_r + GAMMA * tar_value

        if supervised:
            margins = (torch.ones(N_ACTIONS, N_ACTIONS) - torch.eye(N_ACTIONS)) * margin
            batch_margins = margins[b_a.data.squeeze()].to(device)
            q_evals_m = q_evals + batch_margins
            loss_sup = pow((q_evals_m.max(1)[0].unsqueeze(1) - q_eval.detach()), 2)
            loss_sup = loss_sup * b_isdemo[:,np.newaxis]
        if prioritized:
            abs_errors = abs(q_eval.detach().cpu() - q_tar.detach().cpu())

            #print("learn")
            time_new = time()
            self.memory.batch_update(tree_idx, abs_errors)
            #print('update time:%.3f' % (time() - time_new))
            q_eval *= IS ** 0.5
            q_tar *= IS ** 0.5
            if supervised:
                loss_sup *= IS
        if supervised:
            loss_sup = loss_sup.sum() / BATCH_SIZE * lamda
        else:
            loss_sup = torch.Tensor([0]).sum()
        time_new = time()

        loss_td = self.loss_fun(q_eval, q_tar)
        #print('TD:%.4f, SUP:%.4f' %(loss_td, loss_sup))
        loss = loss_td + loss_sup
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print('NN time:%.3f' % (time() - time_new))
        if self.learn_counter % 2 == 0 and online_draw:
                vis.line(X=torch.Tensor
                ([self.learn_counter]), Y=torch.stack((loss_sup.cpu(), loss_td.cpu())).reshape(1, 2),
                         win='Loss', update='append' if self.learn_counter > 0 else None,
                         opts={'title': 'loss', 'legend': ['sup_loss', 'td_loss']})

        # if supervised:
        #     return loss_sup.data[0], loss_td.data[0]
        # else:
        #     return loss.data[0]
        return rate

    def save_memory(self):
        env.end()
        np.savetxt('data/memory/' + memory_save, self.memory[:N_DATA, :], delimiter=',')
        print('\nData generated')
        os._exit(0)

    def load_memory(self, pretrain=False):
        print('Loading memory..................')
        if prioritized:
            memory = np.loadtxt('data/memory/' + memory_load, delimiter=',')
            for i in range(N_DATA):
                self.memory.store(memory[i, :])
            self.memory_counter = N_DATA
        else:
            self.memory[:N_DATA, :] = np.loadtxt('data/memory/' + memory_load,  delimiter=',')
            self.memory_counter = N_DATA
        if pretrain:
            for _ in trange(TIME_PREPRO):
                dqn.learn(True)
            print('PreLearning end')

    def save_model(self):
        torch.save(self.eval_net.state_dict(), 'data/' + method + '/eval_net_CNN.pkl')
        torch.save(self.tar_net.state_dict(), 'data/' + method + '/tar_net_CNN.pkl')
        print('Model saved')

    def load_model(self):
        self.eval_net.load_state_dict(torch.load
                                      ('data/' + method + '/eval_net_CNN.pkl'))
        self.tar_net.load_state_dict(torch.load('data/' + method + '/tar_net_CNN.pkl'))
        print('Model loaded')

    def save_record(self, name, docu):
        if os.path.exists(os.path.join('data', method, para + '_' + name + '_list.csv')):
            docu = np.array(docu).reshape(-1, 1)
            docu_o = np.loadtxt('data/' + method + '/' + para + '_' + name + '_list.csv', delimiter=',')
            docu = np.hstack((docu_o.reshape(docu_o.shape[0], -1), docu))
        np.savetxt('data/' + method + '/' + para + '_' + name + '_list.csv', docu, delimiter=',')
        print(name + ' saved')



args = args_parser()
env = TorcsEnv(port=args.p, text_mode=False, vision=True, throttle=False, gear_change=False)

#env = TorcsEnv(vision=True, throttle=True, gear_change=False)
#vis = visdom.Visdom(env='torcs:' + str(args.p))
console = Console()

if a_replay:
    a_list = np.loadtxt('data/memory/a_list.csv', delimiter=',')


for k in range(1):
    dqn = DQN()
    r_list = []
    ra_list = []
    dist_list = []
    angle_list = []
    loss_list = []
    ratio_list = []
    track_list = []
    loss_sup_list = []
    loss_td_list = []
    track_data_list =[]
    angle_data_list = []
    dist_data_list = []
    r_data_list = []
    print(para + str(k) + " Start")

    if not retrain:
        dqn.load_model()
    if with_data:
        dqn.load_memory(True)

    while (dqn.memory_counter < memory_learn) and learn:
        s_img = np.zeros((IMAGE_NUM, 64, 64))
        s_img_ = np.zeros((IMAGE_NUM, 64, 64))
        s = env.reset()
        s_low = (np.hstack((s.wheelSpinVel / 100.0, s.rpm)))
        img = preprocess(s)

        s_img[0] = img
        s_img_[0] = img

        transitions = []
        a_drive = [0, 0, 0]
        a = 8
        for step in range(9999):
            time_new = time()
            #manul, online_draw, a, record = console.keyboard(a, manul, online_draw)
            if len(s_img) >= IMAGE_NUM:
                if not manul:
                    a = dqn.choose_action(s_img[np.newaxis], s_low[np.newaxis])
                else:
                    dqn.choose_action(s_img[np.newaxis], s_low[np.newaxis])

            a_drive = ACTION[a]
            s_, r, done = env.step(a_drive)
            print('OBSERVE step:%s  reward:%.2f  momery:%s' %(step, r, dqn.memory_counter))
            s_low_ = np.hstack((s_.wheelSpinVel / 100.0, s_.rpm))
            img_ = preprocess(s_)
            s_img_ = up_state(s_img_, img_, step, IMAGE_NUM)
            if step > MAX_STEP:
                done = 1
            if step+1 >= IMAGE_NUM:
                dqn.store_memory(s_img, s_low, a, r, s_img_, s_low_)

            if done:
                break

            s_img = s_img_.copy()
            s_low = s_low_
            print('time:%.3f' % (time() - time_new))
    for i in trange(MAX_EPISODE):
        if dqn.memory_counter >= memory_MAX:
            dqn.adjust_learning_rate(dqn.optimizer, i)

        save_model_flag = False
        if a_record:
            a_list = []

        s_img = np.zeros((IMAGE_NUM, 64, 64))
        s_img_ = np.zeros((IMAGE_NUM, 64, 64))

        s = env.reset()
        s_low = (np.hstack((s.wheelSpinVel / 100.0, s.rpm)))
        img = preprocess(s)

        s_img[0] = img
        s_img_[0] = img

        r_sum = 0
        ratio_sum = 0
        track_sum = 0
        angle_sum = 0

        transitions = []
        a_drive = [0, 0, 0]
        a = 8
        time_now = time()
        time_new = time()
        for step in range(9999):

            time_new = time()
            if dqn.learn_counter % 2 == 0 and online_draw:
                vis.line(X=torch.Tensor([dqn.memory_counter]), Y=torch.Tensor([time_new-time_now]), win='time',
                         update='append' if not dqn.memory_counter%1000 == 0 else None, opts={'title': 'time'})
            time_now = time_new

            manul, online_draw, a, record = console.keyboard(a, manul, online_draw)
            if len(s_img) >= IMAGE_NUM:
                if not manul:
                    a = dqn.choose_action(s_img[np.newaxis], s_low[np.newaxis])
                else:
                    dqn.choose_action(s_img[np.newaxis], s_low[np.newaxis])
            #dqn.action_counter(a) #for action count

            if a_replay:
                a = int(a_list[step])
            if a_record:
                a_list.append(a)

            # a_delta = min(abs(ACTION[a][0] - a_drive[0]), 0.08) * (ACTION[a][0] > a_drive[0])
            # a_drive = a_drive + a_delta
            a_drive = ACTION[a]
            s_, r, done = env.step(a_drive)

            s_low_ = np.hstack((s_.wheelSpinVel / 100.0, s_.rpm))
            img_ = preprocess(s_)
            s_img_ = up_state(s_img_, img_, step, IMAGE_NUM)
            #print('model time:%.3f' % (time() - time_new))
            if record:
                print(step, s_low_)
            if evaluation:
                track_data_list.append(s_.trackPos)
                angle_data_list.append(s_.angle)
                r_data_list.append(r)
            if step > MAX_STEP:
                done = 1

            if step+1 >= IMAGE_NUM:
                #print('save m')
                dqn.store_memory(s_img, s_low, a, r, s_img_, s_low_)

            if dqn.memory_counter >= N_DATA and data_generate:
                np.savetxt('data/' + method + '/human_track.csv', track_data_list, delimiter=',')
                np.savetxt('data/' + method + '/human_angle.csv', angle_data_list, delimiter=',')
                np.savetxt('data/' + method + '/human_r.csv', r_data_list, delimiter=',')
                dqn.save_memory()

            if learn:
                ratio = dqn.learn()
                ratio_sum += ratio

                # if supervised:
                #     loss_sup, loss_td = dqn.learn()
                #     loss_sup_list.append(loss_sup)
                #     loss_td_list.append(loss_td_list)
                # else:
                #     loss = dqn.learn()
                #     loss_list.append(loss)
                if dqn.memory_counter % SAVE_FREQUENCY == 0:
                    save_model_flag = True
            r_sum += r
            angle_sum += abs(s_.angle)
            track_sum += abs(s_.trackPos)

            if dqn.learn_counter % 2 == 0 and online_draw:
                # vis.line(X=torch.Tensor([step]), Y=torch.Tensor([a]), win='a',
                #          update='append' if step > 0 else None, opts={'title': 'a'})

                vis.line(X=torch.Tensor([step]), Y=torch.Tensor([s_.speedX]), win='Xspeed',
                         update='append' if step > 1 else None,  opts={'title': 'Xspeed'})
                vis.line(X=torch.Tensor([step]), Y=torch.Tensor([s_.distRaced]), win='dist_race_step',
                        update='append' if step > 1 else None, opts ={'title': 'dist_race_step'})
                vis.line(X=torch.Tensor([step]), Y=np.array([[r, r1, r2, r3]]), win='r',
                         update='append' if step > 1 else None, opts={'title': 'r','legend': ['r', 'r_long', 'r_late', 'r_track']})
                # if len(s_img) >= IMAGE_NUM:
                #     vis.images(s_img.reshape(IMAGE_NUM, 1, 64, 64), win='image', opts={'title': 'image'})

            if done:
                r_list.append(r_sum/step)
                ra_list.append(r_sum)
                angle_list.append(angle_sum/step)
                dist_list.append(s_.distRaced)
                track_list.append(track_sum/step)
                ratio_list.append(ratio_sum/step)

                if a_record:
                    np.savetxt('data/memory/a_list.csv', np.array(a_list), delimiter=',')
                    print('a_list saved')
                print('\n\nep:%s  av-reward:%.3f  step:%s  memory:%s' %(i, r_sum/step, step, dqn.memory_counter))
                #print('\n---------------------------\nstep', step, 'reward', r_sum/step, 'memory', dqn.memory_counter)
                if online_draw:
                    vis.line(X=torch.Tensor([i]), Y=torch.Tensor([(r_sum)/step]), win='r_average',
                             update='append' if dqn.memory_counter > 0 else None,  opts={'title': 'R_average'})
                    vis.line(X=torch.Tensor([i]), Y=torch.Tensor([(r_sum)]), win='r_sum',
                             update='append' if dqn.memory_counter > 0 else None, opts={'title': 'R_sum'})
                    vis.line(X=torch.Tensor([i]), Y=torch.Tensor([s_.distRaced]), win='dist_race',
                             update='append' if dqn.memory_counter > 0 else None, opts={'title': 'dist_race'})
                if save_model_flag == True:
                    dqn.save_model()
                break
            #sleep(0.1)
            s_img = s_img_.copy()
            s_low = s_low_
            #print('time:%.3f' %(time()-time_new))
            #print('step:%s  reward:%.2f  momery:%s  time:%.3f' % (step, r, dqn.memory_counter, time()-time_new))
            #print(step_loop.set_description('step:%s  reward:%.2f' % (step, r))
            #print("data", dqn.memory_counter, 'reward', r)

    dqn.save_record('r', r_list)
    dqn.save_record('ra', ra_list)
    dqn.save_record('dist', dist_list)
    dqn.save_record('angle', angle_list)
    dqn.save_record('track', track_list)
    dqn.save_record('ratio', r_list)


    # if supervised:
    #     if os.path.exists(os.path.join('data', method, para + '_loss_sup_list.csv')):
    #         loss_sup_list = np.array(loss_sup_list).reshape(-1, 1)
    #         loss_sup_list_o = np.loadtxt('data/' + method + '/' + para + '_loss_sup_list.csv', delimiter=',')
    #         loss_sup_list = np.hstack((loss_sup_list_o.reshape(loss_sup_list_o.shape[0], -1), loss_sup_list))
    #     if os.path.exists(os.path.join('data', method, para + '_loss_td_list.csv')):
    #         loss_td_list = np.array(loss_td_list).reshape(-1, 1)
    #         loss_td_list_o = np.loadtxt('data/' + method + '/' + para + '_loss_td_list.csv', delimiter=',')
    #         loss_td_list = np.hstack((loss_td_list_o.reshape(loss_td_list_o.shape[0], -1), loss_td_list))
    #     np.savetxt('data/' + method + '/' + para + '_loss_sup_list.csv', loss_sup_list, delimiter=',')
    #     np.savetxt('data/' + method + '/' + para + '_loss_td_list.csv', loss_td_list, delimiter=',')
    #
    # else:
    #     if os.path.exists(os.path.join('data', method, para + '_loss_list.csv')):
    #         loss_list = np.array(loss_list).reshape(-1, 1)
    #         loss_list_o = np.loadtxt('data/' + method + '/' + para + '_loss_list.csv', delimiter=',')
    #         loss_list = np.hstack((loss_list_o.reshape(loss_list_o.shape[0], -1), loss_list))
    #     np.savetxt('data/' + method + '/' + para + '_loss_list.csv', loss_list, delimiter=',')


env.end()
