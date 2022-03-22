import os
from collections import OrderedDict
from copy import deepcopy
from typing import Union, Dict, List, Callable
import threading
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from rgb_stacking.utils.mpi_tools import msg

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
                  # device,
                  use_multi_thread):

    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    # envs = MultiThreadedVecEnv(envs) if use_multi_thread else DummyVecEnv(envs)
    envs = SubprocVecEnv(envs) if use_multi_thread else DummyVecEnv(envs)
    envs = VecNormalize(envs, gamma=gamma)

    # envs = VecPyTorch(envs, device)

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


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


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


class Worker(threading.Thread):
    def __init__(self, env, venv, env_idx):
        super().__init__()
        self.env = env
        self.venv = venv
        self.env_idx = env_idx

    def run(self) -> None:
        while not self.venv.done:
            self.venv.begin_barrier.wait()
            obs, self.venv.buf_rews[self.env_idx], \
            self.venv.buf_dones[self.env_idx], \
            self.venv.buf_infos[self.env_idx] \
                = self.venv[self.env_idx].step(self.venv.actions[self.env_idx])
            if self.venv.buf_dones[self.env_idx]:
                # save final observation where user can get it, then reset
                self.venv.buf_infos[self.env_idx]["terminal_observation"] = obs
                obs = self.venv.envs[self.env_idx].reset()
            self.venv._save_obs(self.env_idx, obs)
            self.venv.end_barrier.wait()


class MultiThreadedVecEnv(DummyVecEnv):

    def _complete_step(self):
        self.step_return = self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        msg('Creating Threads')
        self.done = False
        self.step_return = None

        self.begin_barrier, self.end_barrier = threading.Barrier(self.num_envs + 1), \
                                               threading.Barrier(self.num_envs + 1, self._complete_step)

        self.threads = [Worker(env, self, i) for i, env in enumerate(self.envs)]
        for t in self.threads:
            t.start()

    def step_wait(self) -> VecEnvStepReturn:
        self.begin_barrier.wait()
        self.end_barrier.wait()
        return self.step_return

    def __getitem__(self, index):
        return self.envs[index]

    def close(self) -> None:
        msg('joining all threads')
        self.done = True
        for t in self.threads:
            t.join()


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def revert_action_reward(self, obs_dict: Dict):
        original_obs = super().get_original_obs()
        obs_dict['past_reward'] = original_obs['past_reward']
        obs_dict['past_action'] = original_obs['past_action']
        return obs_dict

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = super().reset()
        return self.revert_action_reward(obs)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rew, done, info = super(VecNormalize, self).step_wait()
        step_ret_ = self.revert_action_reward(obs), rew, done, info
        return step_ret_

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
