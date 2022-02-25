from abc import ABC
from typing import Dict, Sequence, Optional

import cv2
import gym
import numpy
import numpy as np

from rgb_stacking import environment


def flatten_dict(obs: Dict, keys):
    return np.concatenate([np.ravel(obs[k]) for k in keys])


class GymWrapper(gym.Env):

    def make_box_obs_space(self, obs_spec):
        sz = 0
        for k, v in obs_spec.items():
            self.flatten_order.append(k)
            sz += np.prod(v.shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, [sz])

    def make_dict_obs_space(self, obs_spec):
        shape_map = dict()
        for k, v in obs_spec.items():
            shape_map[k] = gym.spaces.Box(-np.inf, np.inf, [np.prod(v.shape)])
        self.observation_space = gym.spaces.Dict(shape_map)

    def make_discrete_action_space(self, num_discrete_action_bin):
        assert num_discrete_action_bin % 2 == 1, 'number of discrete action bin must be odd'

        BINARY_GRIPPER_IDX = -1
        bins = [num_discrete_action_bin for _ in range(self.action_spec.shape[0])]
        bins[BINARY_GRIPPER_IDX] = 2

        self.action_space = gym.spaces.MultiDiscrete(bins)
        self.discrete_action_bin = [np.linspace(_min, _max, num_discrete_action_bin)
                                    for _min, _max in zip(self.action_spec.minimum[:-1], self.action_spec.maximum[:-1])]
        self.discrete_action_bin.append(np.array([self.action_spec.minimum[-1], self.action_spec.maximum[-1]]))

    def make_continuous_action_space(self):
        self.action_space = gym.spaces.Box(-1, 1, self.action_spec.shape)

    def __init__(self, object_triplet,
                 flatten_observation_space=True,
                 num_discrete_action_bin: int = None):

        self.env = environment.rgb_stacking(object_triplet=object_triplet)
        self.flatten_observation_space = flatten_observation_space

        obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        self.discrete_action_bin = None
        self.flatten_order = []

        if flatten_observation_space:
            self.make_box_obs_space(obs_spec)
        else:
            self.make_dict_obs_space(obs_spec)

        if num_discrete_action_bin:
            self.make_discrete_action_space(num_discrete_action_bin)
        else:
            self.make_continuous_action_space()

    def observation(self, obs):
        return flatten_dict(obs, self.flatten_order) if self.flatten_observation_space \
            else {k: np.ravel(v) for k, v in obs.items()}

    def reset(self):
        return self.observation(self.env.reset().observation)

    def render(self, mode="human"):
        cam = self.env.physics.render(camera_id='main_camera', width=480, height=240)
        cv2.imshow('Main Camera', cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def step(self, action: numpy.ndarray):

        action = np.array([bins[action_index]
                           for bins, action_index in
                           zip(self.discrete_action_bin, action)]) if self.discrete_action_bin \
            else np.array(
            [norm_action * scale_factor for scale_factor, norm_action in zip(self.action_spec.maximum, action)])

        time_step = self.env.step(action)

        return self.observation(time_step.observation), time_step.reward, time_step.last(), {}

    def close(self):
        self.env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    import tabulate

    env = GymWrapper('rgb_train_random', False, 21)

    print('observation_space\n', tabulate.tabulate([ [k, v.shape] for k, v in env.observation_space.spaces.items()]))
    print('action_space', env.action_space)

    reset_obs = env.reset()
    print(tabulate.tabulate([[k, v] for k, v in reset_obs.items()]))

    action = env.action_space.sample()
    print('take random action', action)
    print(tabulate.tabulate([[k, v] for k, v in env.step(action)[0].items()]))
    # print(env.step(action))

    env.close()


if __name__ == '__main__':
    from absl import app

    app.run(main)
