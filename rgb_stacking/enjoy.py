import argparse
import os
# workaround to unpickle olf model files

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from rgb_stacking.contrib.envs import make_vec_envs, VecPyTorch
from rgb_stacking.utils.utils import get_render_func, get_vec_normalize
from dm_control import viewer
# sys.path.append('a2c_ppo_acktr')


parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = \
    torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
               map_location='cpu')

print(actor_critic)

if str(args.env_name).startswith('StackRGBTrain-v1'):
    envs = ['StackRGBTestTriplet-v{}'.format(i) for i in range(5) ]
else:
    envs = ['StackRGBTestTripletActorDict-v{}'.format(i) for i in range(5)]

for env_name in envs:
    env = VecPyTorch(make_vec_envs(
        env_name,
        args.seed + 1000,
        1,
        None,
        False), device='cpu')
    # Get a render function
    # render_func = get_render_func(env)

    returns = 0
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
        DummyVecEnv.action_spec = lambda self: env.action_spec()
        DummyVecEnv.physics = property(lambda self: env.physics)

    recurrent_hidden_states = actor_critic.zero_state(1, "lstm")
    masks = torch.zeros(1, 1)

    def policy_(timestep):
        global recurrent_hidden_states, returns
        returns += timestep[1]
        done = timestep[2]
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                timestep[0], recurrent_hidden_states, masks, deterministic=args.det)
            masks.fill_(0.0 if done else 1.0)
            return action

    viewer.launch(env, policy_)

    print('Total Returns: ', returns)
    env.close()