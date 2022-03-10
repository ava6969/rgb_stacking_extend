import enum
from abc import ABC
from typing import Dict, Sequence, Optional

import cv2
import gym
import numpy
import numpy as np

from rgb_stacking import environment


def flatten_dict(obs: Dict, keys):
    return np.concatenate([np.ravel(obs[k]) for k in keys])


class ObservationPreprocess(enum.Enum):
    RAW_DICT = 1
    FLATTEN = 2
    ACTOR_BASED = 3


NUM_OF_JOINTS_ON_ARMS = 7


def actor_based_observation(obs):
    obs_dict = {'robot0:joint' + str(i):
                    np.concatenate([np.ravel(obs['sawyer/joints/angle'][:, i]),
                                    np.ravel(obs['sawyer/joints/torque'][:, i]),
                                    np.ravel(obs['sawyer/joints/velocity'][:, i]),
                                    np.ravel(obs['action/environment'][:, i])])
                for i in range(NUM_OF_JOINTS_ON_ARMS)}

    obs_dict['robot0:wrist'] = np.concatenate([np.ravel(obs['wrist/force']),
                                               np.ravel(obs['wrist/torque']),
                                               np.ravel(obs['sawyer/tcp/pose']),
                                               np.ravel(obs['sawyer/tcp/velocity'])])
    obs_dict['robot0:pinch'] = np.ravel(obs['sawyer/pinch/pose'])
    obs_dict['robot0:gripper'] = np.ravel(np.array([obs['gripper/grasp'],
                                                    obs['gripper/joints/angle'],
                                                    obs['gripper/joints/velocity']]))
    for color in ['red', 'blue', 'green']:
        obs_dict['{}:pos_relative_to_pinch+pose'.format(color)] = np.ravel(np.array(
            [obs['rgb30_{}/abs_pose'.format(color)], obs['rgb30_{}/to_pinch'.format(color)]]))
    return obs_dict


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

    def make_actor_based_obs_space(self, obs_spec):
        space_fn = lambda n: gym.spaces.Box(-np.inf, np.inf, [n])
        flatten = lambda k: np.prod(obs_spec[k].shape)

        flatten_arm_joint_size = np.sum([flatten('sawyer/joints/angle'),
                                         flatten('sawyer/joints/torque'),
                                         flatten('sawyer/joints/velocity'),
                                         flatten('action/environment')])
        size_per_arm_joint = flatten_arm_joint_size // NUM_OF_JOINTS_ON_ARMS

        shape_map = {'robot0:joint' + str(i): space_fn(size_per_arm_joint)
                     for i in range(NUM_OF_JOINTS_ON_ARMS)}

        shape_map['robot0:wrist'] = space_fn(flatten('wrist/force') + flatten('wrist/torque') +
                                             flatten('sawyer/tcp/pose') + flatten('sawyer/tcp/velocity'))

        shape_map['robot0:pinch'] = space_fn(flatten('sawyer/pinch/pose'))
        shape_map['robot0:gripper'] = space_fn(flatten('gripper/grasp') +
                                               flatten('gripper/joints/angle') +
                                               flatten('gripper/joints/velocity'))

        rgb_size = np.prod(obs_spec['rgb30_blue/abs_pose'].shape) + np.prod(obs_spec['rgb30_blue/to_pinch'].shape)
        shape_map['red:pos_relative_to_pinch+pose'] = space_fn(rgb_size)
        shape_map['blue:pos_relative_to_pinch+pose'] = space_fn(rgb_size)
        shape_map['green:pos_relative_to_pinch+pose'] = space_fn(rgb_size)

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
                 obs_preprocess=ObservationPreprocess.FLATTEN,
                 num_discrete_action_bin: int = None):

        self.env = environment.rgb_stacking(object_triplet=object_triplet)
        self.obs_preprocess = obs_preprocess

        obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        self.discrete_action_bin = None
        self.flatten_order = []

        if obs_preprocess.value == ObservationPreprocess.FLATTEN.value:
            self.make_box_obs_space(obs_spec)
        elif obs_preprocess.value == ObservationPreprocess.RAW_DICT.value:
            self.make_dict_obs_space(obs_spec)
        else:
            self.make_actor_based_obs_space(obs_spec)

        if num_discrete_action_bin:
            self.make_discrete_action_space(num_discrete_action_bin)
        else:
            self.make_continuous_action_space()

    def observation(self, obs):
        return flatten_dict(obs, self.flatten_order) if self.obs_preprocess.value == ObservationPreprocess.FLATTEN.value \
            else {k: np.ravel(v) for k, v in obs.items()} if self.obs_preprocess.value == ObservationPreprocess.RAW_DICT.value \
            else actor_based_observation(obs)

    def reset(self):
        success_obs = None
        while not success_obs:
            try:
                success_obs = self.env.reset()
            except Exception:
                success_obs = None

        return self.observation(success_obs.observation)

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

    env = GymWrapper('rgb_train_random', ObservationPreprocess.ACTOR_BASED, 11)
    print(env.observation_space.spaces.keys())
    print('observation_space\n', tabulate.tabulate([[k, v.shape] for k, v in env.observation_space.spaces.items()]))
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
