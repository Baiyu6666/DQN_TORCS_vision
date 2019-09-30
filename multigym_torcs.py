import gym
from gym import spaces
from baselines import logger
import numpy as np
import subprocess
import multisnakeoil3_gym as snakeoil3
import numpy as np
import signal
import copy
import collections as col
import os
import math
import time


class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 1  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 80
    limit_reward_l = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False, trackname='unknown', port=3101, text_mode=True):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.trackname = trackname.strip()
        self.text_mode = text_mode
        self._max_episode_steps = 1500
        if self.trackname is not "unknown":
            assert '*' in self.trackname

        self.initial_run = True
        self.lap = 0
        self.reset_count = 0
        self.prev_distance_from_start = 1

        self.port = port
        self.torcs_proc = None
        time.sleep(0.5)
        self.start_torcs_process()
        time.sleep(0.5)

        # The first version is with brak
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

        if vision is False:
            
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29, ))
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29, ))
    
    def start_torcs_process(self):
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None
        window_title = str(self.port)
        torcs_cmd = ('torcs')
        if self.text_mode:
            torcs_cmd = torcs_cmd + ' -T'
        command = '{} -nofuel -nodamage -nolaptime -title {} -p {}'.format(torcs_cmd, window_title, self.port)
        if self.vision is True:
            command += ' -vision'
        if '*' in self.trackname:
            command += ' -tr %s'%(self.trackname)
        logger.log(command)
        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(0.5)
        if '-T' not in command:
            os.system('sh autostart.sh {}'.format(window_title))
        time.sleep(0.1)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50*2):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.3:
                client.R.d['accel'] = 0.3

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        if self.prev_distance_from_start > self.observation.distFromStart and abs(self.observation.angle)<.1:
            self.lap+= 1
        self.prev_distance_from_start = self.observation.distFromStart

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])

        reward_long = sp *np.cos(obs['angle']) / self.default_speed
        reward_late = - np.abs(sp*np.sin(obs['angle'])) / self.default_speed
        reward_track = - sp * np.abs(obs['trackPos']) / self.default_speed
        #print(reward_track)
        reward = reward_long + reward_late + reward_track

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -2

        # Here we did not reset again for running one lap
        #TODO we may add this section of codes later
        # Termination judgement #########################
        episode_terminate = False
        
        if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
           print('Out of Track')
           reward = -1
           #episode_terminate = True
           client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
           if progress < (self.termination_limit_progress / self.default_speed):
               reward = -1
               print("No progress")
               episode_terminate = True
               #TODO check if we really need this one
               client.R.d['meta'] = True
        
        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        #TODO firstly we try without stop verision
        # later we try a better version that reset

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        return self.get_obs(), reward, client.R.d['meta'], reward_long, reward_late, reward_track

    def reset(self, relaunch=True):
        #print("Reset")
        self.reset_count += 1
        if np.mod(self.reset_count, 50) == 0:
            relaunch = True
        else:
            relaunch = False

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.prev_distance_from_start = 1
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                self.lap = 0
                logger.log("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        # self.client = snakeoil3.Client(p=3101, vision=self.vision, t=self.trackname)  # Open new UDP in vtorcs
        self.client = snakeoil3.Client(self.start_torcs_process, p=self.port)
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        # os.system('pkill torcs')
         os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
    
    def get_obs(self):
        ob = self.observation
        return ob

    def reset_torcs(self):
       #print("relaunch torcs")
       
       self.torcs_proc.terminate()
       time.sleep(0.5)
       self.start_torcs_process()
       time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)
    
    def log_envs(self):
        logger.record_tabular('lap', self.lap)
        logger.record_tabular('prev_distance_from_start', self.prev_distance_from_start)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel',
                     'distRaced',
                     'distFromStart']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/math.pi,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               distRaced=np.array(raw_obs['distRaced'], dtype=np.float32) / 1.,
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32) / 1.)
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img',
                     'distRaced',
                     'distFromStart'
                     ]
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[10]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               angle=np.array(raw_obs['angle'], dtype=np.float32) / math.pi,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb,
                               distRaced=np.array(raw_obs['distRaced'], dtype=np.float32) / 1.,
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32) / 1.)
