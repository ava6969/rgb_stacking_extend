from abc import ABC
from typing import Dict, Sequence, Optional

import gym
import numpy
import numpy as np

from rgb_stacking import environment


def flatten_dict(obs: Dict, keys):
    return np.concatenate([np.ravel(obs[k]) for k in keys])


class GymWrapper(gym.Env):

    def __init__(self, object_triplet,
                 flatten=True,
                 discrete_n: int = None):
        self.env = environment.rgb_stacking(object_triplet=object_triplet)
        self.flatten = flatten

        obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()
        self.action_bins = None

        self.flatten_order = []
        if flatten:
            sz = 0
            for k, v in obs_spec.items():
                self.flatten_order.append(k)
                sz += np.prod(v.shape)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, [sz])
        else:
            shape_map = dict()
            for k, v in obs_spec.items():
                shape_map[k] = gym.spaces.Box(-np.inf, np.inf, [np.prod(v.shape)])
            self.observation_space = gym.spaces.Dict(shape_map)

        if discrete_n:
            assert(discrete_n % 2 == 1, 'discrete_n must be odd')
            bins = [discrete_n for _ in range(self.action_spec.shape[0])]
            bins[-1] = 2
            self.action_space = gym.spaces.MultiDiscrete(bins)
            self.action_bins = [np.linspace(_min, _max, discrete_n)
                                for _min, _max in zip(self.action_spec.minimum[:-1], self.action_spec.maximum[:-1])]
            self.action_bins.append(np.array([self.action_spec.minimum[-1], self.action_spec.maximum[-1]]))
        else:
            self.action_space = gym.spaces.Box(-1, 1, self.action_spec.shape)

    def observation(self, obs):
        return flatten_dict(obs, self.flatten_order) if self.flatten else {k: np.ravel(v) for k, v in obs.items()}

    def reset(self):
        return self.observation(self.env.reset().observation)

    def step(self, action: numpy.ndarray):
        action = np.array([bins[a] for bins, a in zip(self.action_bins, action)])\
            if self.action_bins\
            else np.array([a*factor for factor, a in zip(self.action_spec.maximum, action)])
        time_step = self.env.step(action)
        return self.observation(time_step.observation), time_step.reward, time_step.last(), {}

    def close(self):
        self.env.close()


def main(argv: Sequence[str]) -> None:
    del argv

    env = GymWrapper('rgb_train_random', False, 21)

    print('observation_space', env.observation_space)
    print('action_space', env.action_space)

    print(env.reset())
    print(env.step(env.action_space.sample()))

    env.close()

if __name__ == '__main__':
    from absl import app
    app.run(main)

