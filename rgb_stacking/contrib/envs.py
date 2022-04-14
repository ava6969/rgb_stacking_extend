from collections import OrderedDict
import gym
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
            env = Monitor(env, None, allow_early_resets=False)
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  use_multi_thread):

    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    envs = SubprocVecEnv(envs, "forkserver") if use_multi_thread else DummyVecEnv(envs)

    envs = VecNormalize(envs, gamma=gamma)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def observation(self, obs):
        if isinstance(obs, OrderedDict):
            return {k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()}
        else:
            return torch.from_numpy(obs).float().to(self.device)

    def reset(self):
        return self.observation(self.venv.reset())

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.observation(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
