import tensorflow as tf 
import os.path as osp
import numpy as np 
import sys
import argparse
from baselines import logger
from datetime import datetime
from baselines.common.cmd_util import set_global_seeds
from baselines.common import tf_util as U 
from multigym_torcs import TorcsEnv

def train(road_type, num_timesteps, seed, port):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, 
                ac_space=ac_space, hid_size=64, num_hid_layers=2)
    set_global_seeds(seed)
    # environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, port=port, trackname=road_type)

    try:
        pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear')
        env.end()
    except KeyboardInterrupt:
        print("Iterrupted")
        env.end()
        sys.exit(0)
    
    
def torcs_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--road_type', help='road type', default='g-track-1*road')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(1e7))
    parser.add_argument('--p', type=int, default=3101)

    args = parser.parse_args()
    
    return args


def main():
    args = torcs_args_parser()
    if args.p == 3101:
        args.p = np.random.randint(low=10000, high=65000)
        logger.log("port with %d"%(args.p))
    
    logdir='torcs_log/cat=%s/road=%s/seed=%d_%s'%(
        args.road_type.split('*')[1],
        args.road_type,
        args.seed,
        datetime.now().strftime('%d_%H:%M:%S'))

    logger.configure(logdir)

   
    train(road_type=args.road_type,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        port=args.p)


if __name__ == '__main__':
    main()
    
