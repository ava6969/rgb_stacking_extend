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
    joint_sensor = [np.concatenate([np.ravel(obs['sawyer/joints/angle'][:, i]),
                                    np.ravel(obs['sawyer/joints/torque'][:, i]),
                                    np.ravel(obs['sawyer/joints/velocity'][:, i]),
                                    np.ravel(obs['action/environment'][:, i])])
                    for i in range(NUM_OF_JOINTS_ON_ARMS)]

    obs_dict = dict({'robot0:joints': np.concatenate(joint_sensor)})

    obs_dict['robot0:others'] = np.concatenate([np.ravel(obs['wrist/force']),
                                                np.ravel(obs['wrist/torque']),
                                                np.ravel(obs['sawyer/tcp/pose']),
                                                np.ravel(obs['sawyer/tcp/velocity']),
                                                np.ravel(obs['sawyer/pinch/pose']),
                                                np.ravel(np.array([obs['gripper/grasp'],
                                                                   obs['gripper/joints/angle'],
                                                                   obs['gripper/joints/velocity']]))
                                                ])

    boxes = [np.ravel(np.array([obs['rgb30_{}/abs_pose'.format(color)],
                                obs['rgb30_{}/to_pinch'.format(color)]])) for color in ['red',
                                                                                        'blue',
                                                                                        'green']]
    obs_dict['boxes'] = np.concatenate(boxes)

    return obs_dict


def box_space(n):
    return gym.spaces.Box(-np.inf, np.inf, [n])


class GymWrapper(gym.Env):
    ACTION_BIN_SIZE = 11

    def make_space(self, obs_spec, shape_map):
        shape_map['past_reward'] = gym.spaces.Box(-np.inf, np.inf, [1])
        shape_map['past_action'] = gym.spaces.Box(np.array([0, 0, 0, 0, 0]),
                                                  np.array([GymWrapper.ACTION_BIN_SIZE - 1]*4 + [1]))
        if self.add_image:
            img_shape = np.roll(obs_spec['basket_back_left/pixels'].shape, 1)
            shape_map['image_bl'] = gym.spaces.Box(0, 1, img_shape)
            shape_map['image_fl'] = gym.spaces.Box(0, 1, img_shape)
            shape_map['image_fr'] = gym.spaces.Box(0, 1, img_shape)

        self.observation_space = gym.spaces.Dict(shape_map)

    def make_box_obs_space(self, obs_spec):
        sz = 0
        for k, v in obs_spec.items():
            self.flatten_order.append(k)
            sz += np.prod(v.shape)
        self.make_space(obs_spec, dict(observation=gym.spaces.Box(-np.inf, np.inf, [sz])))

    def make_dict_obs_space(self, obs_spec):
        shape_map = dict()
        for k, v in obs_spec.items():
            shape_map[k] = gym.spaces.Box(-np.inf, np.inf, [np.prod(v.shape)])
        self.make_space(obs_spec, shape_map)

    def make_actor_based_obs_space(self, obs_spec):
        flatten = lambda k: np.prod(obs_spec[k].shape)

        flatten_arm_joint_size = np.sum([flatten('sawyer/joints/angle'),
                                         flatten('sawyer/joints/torque'),
                                         flatten('sawyer/joints/velocity'),
                                         flatten('action/environment')])

        shape_map = dict({'robot0:joints': box_space(flatten_arm_joint_size - 2)})
        shape_map['robot0:others'] = box_space(flatten('wrist/force') + flatten('wrist/torque') +
                                               flatten('sawyer/tcp/pose') + flatten('sawyer/tcp/velocity')
                                               + flatten('sawyer/pinch/pose') + flatten('gripper/grasp') +
                                               flatten('gripper/joints/angle') +
                                               flatten('gripper/joints/velocity'))

        rgb_size = np.prod(obs_spec['rgb30_blue/abs_pose'].shape) + np.prod(obs_spec['rgb30_blue/to_pinch'].shape)
        shape_map['boxes'] = box_space(rgb_size * 3)
        self.make_space(obs_spec, shape_map)

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
                 num_discrete_action_bin: int = None,
                 add_image=False):

        GymWrapper.ACTION_BIN_SIZE = num_discrete_action_bin
        self.env = environment.rgb_stacking(object_triplet=object_triplet,
                                            observation_set=environment.ObservationSet.ALL if add_image
                                            else environment.ObservationSet.STATE_ONLY)
        self.add_image = add_image
        self.obs_preprocess = obs_preprocess

        obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        self.discrete_action_bin = None
        self.flatten_order = []
        self.past_action = [num_discrete_action_bin // 2 for _ in range(self.action_spec.shape[0] - 1)] + [0]
        self.past_reward = 0

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
        base = {'observation': flatten_dict(obs, self.flatten_order)} \
            if self.obs_preprocess.value == ObservationPreprocess.FLATTEN.value \
            else {k: np.ravel(v) for k, v in
                  obs.items()} if self.obs_preprocess.value == ObservationPreprocess.RAW_DICT.value \
            else actor_based_observation(obs)
        base['past_reward'] = np.array([self.past_reward], float)
        base['past_action'] = np.array(self.past_action, int)
        if self.add_image:
            # cv2.imshow('image_bl', cv2.cvtColor(obs['basket_back_left/pixels'], cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
            # cv2.imshow('image_fl', cv2.cvtColor(obs['basket_front_left/pixels'], cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
            # cv2.imshow('image_fr', cv2.cvtColor(obs['basket_front_right/pixels'], cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)
            base['image_bl'] = np.transpose(obs['basket_back_left/pixels'], (2, 0, 1))/255
            base['image_fl'] = np.transpose(obs['basket_front_left/pixels'], (2, 0, 1))/255
            base['image_fr'] = np.transpose(obs['basket_front_right/pixels'], (2, 0, 1))/255
        return base

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
        _clone_action = action.copy()
        np_action = np.array([bins[action_index]
                              for bins, action_index in
                              zip(self.discrete_action_bin, action)]) if self.discrete_action_bin \
            else np.array(
            [norm_action * scale_factor for scale_factor, norm_action in zip(self.action_spec.maximum, action)])

        time_step = self.env.step(np_action)
        x = self.observation(time_step.observation)
        self.past_action = _clone_action
        self.past_reward = time_step.reward
        return x, time_step.reward, time_step.last(), {}

    def close(self):
        self.env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    import tabulate

    env = GymWrapper('rgb_train_random', ObservationPreprocess.ACTOR_BASED, 11, True)
    print(env.observation_space.spaces.keys())
    print('observation_space\n', tabulate.tabulate([[k, v.shape] for k, v in env.observation_space.spaces.items()]))
    print('action_space', env.action_space)

    reset_obs = env.reset()
    print(tabulate.tabulate([[k, v] for k, v in reset_obs.items()]))

    action = env.action_space.sample()
    print('take random action', action)
    print(tabulate.tabulate([[k, v] for k, v in env.step(action)[0].items()]))
    # print(env.step(action))

    action = env.action_space.sample()
    print('take random action', action)
    print(tabulate.tabulate([[k, v] for k, v in env.step(action)[0].items()]))
    # print(env.step(action))

    env.close()


if __name__ == '__main__':
    from absl import app

    app.run(main)
