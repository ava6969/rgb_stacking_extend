import enum
from typing import Dict, Sequence

import cv2
import gym
import numpy
import numpy as np

from rgb_stacking import environment


class ObservationPreprocess(enum.Enum):
    RAW_DICT = 1
    FLATTEN = 2
    ACTOR_BASED = 3


NUM_OF_JOINTS_ON_ARMS = 7


def exclude(key, policy):
    return (policy and key == 'wrist/force') or (policy and key == 'wrist/torque') or key == 'action/environment'


def flatten_dict(obs: Dict, keys, policy: bool):
    return np.concatenate([np.ravel(obs[k]) for k in keys if not exclude(k, policy)])


def actor_based_observation(obs, policy):
    header = 'actor::' if policy else 'critic::'
    joint_sensor = [np.concatenate([np.ravel(obs['sawyer/joints/angle'][:, i]),
                                    np.ravel(obs['sawyer/joints/torque'][:, i]),
                                    np.ravel(obs['sawyer/joints/velocity'][:, i])])
                    for i in range(NUM_OF_JOINTS_ON_ARMS)]

    obs_dict = dict({header + 'joints': np.concatenate(joint_sensor)})

    obs_dict[header + 'wrist'] = np.concatenate(([np.ravel(obs['wrist/force']),
                                                  np.ravel(obs['wrist/torque'])] if not policy else []) +
                                                [np.ravel(obs['sawyer/tcp/pose']),
                                                 np.ravel(obs['sawyer/tcp/velocity'])])

    obs_dict[header + 'gripper'] = np.concatenate([np.ravel(obs['sawyer/pinch/pose']),
                                                   np.ravel(np.array([obs['gripper/grasp'], obs['gripper/joints/angle'],
                                                                      obs['gripper/joints/velocity']]))])

    obs_dict[header + 'boxes'] = np.concatenate([np.ravel(np.array([obs['rgb30_{}/abs_pose'.format(color)],
                                                                    obs['rgb30_{}/to_pinch'.format(color)]])) for color
                                                 in ['red', 'blue', 'green']])

    return obs_dict


def box_space(n):
    return gym.spaces.Box(-np.inf, np.inf, [n])


class GymWrapper(gym.Env):
    ACTION_BIN_SIZE = 11

    def make_space(self, obs_spec, shape_map):
        if self.add_image:
            img_shape = np.roll(obs_spec['basket_back_left/pixels'].shape, 1)
            shape_map['image_bl'] = gym.spaces.Box(0, 1, img_shape)
            shape_map['image_fl'] = gym.spaces.Box(0, 1, img_shape)
            shape_map['image_fr'] = gym.spaces.Box(0, 1, img_shape)

        shape_map['past_action'] = box_space(np.prod(obs_spec['action/environment'].shape))

        self.observation_space = gym.spaces.Dict(shape_map)

    def make_box_obs_space(self, obs_spec, policy: bool):
        sz = 0
        self.flatten_order['actor' if policy else 'critic'] = []
        for k, v in obs_spec.items():
            if exclude(k, policy):
                continue
            self.flatten_order['actor' if policy else 'critic'].append(k)
            sz += np.prod(v.shape)
        return gym.spaces.Box(-np.inf, np.inf, [sz])

    def make_actor_based_obs_space(self, obs_spec, policy: bool):
        flatten = lambda k: np.prod(obs_spec[k].shape)
        header = 'actor::' if policy else 'critic::'

        flatten_arm_joint_size = np.sum([flatten('sawyer/joints/angle'),
                                         flatten('sawyer/joints/torque'),
                                         flatten('sawyer/joints/velocity')])

        shape_map = dict({header + 'joints': box_space(flatten_arm_joint_size)})
        wrist_flat_sz = flatten('wrist/force') + flatten('wrist/torque') if not policy else 0

        shape_map[header + 'wrist'] = box_space(
            wrist_flat_sz + flatten('sawyer/tcp/pose') + flatten('sawyer/tcp/velocity'))

        shape_map[header + 'gripper'] = box_space(flatten('sawyer/pinch/pose') + flatten('gripper/grasp') +
                                                  flatten('gripper/joints/angle') +
                                                  flatten('gripper/joints/velocity'))

        rgb_size = np.prod(obs_spec['rgb30_blue/abs_pose'].shape) + np.prod(obs_spec['rgb30_blue/to_pinch'].shape)
        shape_map[header + 'boxes'] = box_space(rgb_size * 3)

        return shape_map

    def make_discrete_action_space(self, num_discrete_action_bin):
        assert num_discrete_action_bin % 2 == 1, 'number of discrete action bin must be odd'

        BINARY_GRIPPER_IDX = -1
        bins = [num_discrete_action_bin for _ in range(self.action_spec.shape[0])]
        bins[BINARY_GRIPPER_IDX] = 2

        self.action_space = gym.spaces.MultiDiscrete(bins)
        self.discrete_action_bin = [np.linspace(_min, _max, num_discrete_action_bin)
                                    for _min, _max in zip(self.action_spec.minimum[:-1], self.action_spec.maximum[:-1])]
        self.discrete_action_bin.append(np.array([self.action_spec.minimum[-1], self.action_spec.maximum[-1]]))

    def __init__(self, object_triplet,
                 obs_preprocess=ObservationPreprocess.FLATTEN,
                 num_discrete_action_bin: int = None,
                 add_image=False,
                 domain_random=False):

        self.domain_random = domain_random


        GymWrapper.ACTION_BIN_SIZE = num_discrete_action_bin
        self.env = environment.rgb_stacking(object_triplet=object_triplet,
                                            observation_set=environment.ObservationSet.ALL if add_image
                                            else environment.ObservationSet.STATE_ONLY)

        self.add_image = add_image
        self.obs_preprocess = obs_preprocess

        obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        self.discrete_action_bin = None
        self.flatten_order = dict()

        if obs_preprocess.value == ObservationPreprocess.FLATTEN.value:
            space_map = dict(actor=self.make_box_obs_space(obs_spec, True),
                             critic=self.make_box_obs_space(obs_spec, False))
        else:
            space_map = self.make_actor_based_obs_space(obs_spec, True)
            space_map.update(self.make_actor_based_obs_space(obs_spec, False))

        self.make_space(obs_spec, space_map)
        self.make_discrete_action_space(num_discrete_action_bin)

    def _observation(self, obs):

        if self.add_image:
            return {'image_bl': np.transpose(obs['basket_back_left/pixels'], (2, 0, 1)) / 255,
                    'image_fl': np.transpose(obs['basket_front_left/pixels'], (2, 0, 1)) / 255,
                    'image_fr': np.transpose(obs['basket_front_right/pixels'], (2, 0, 1)) / 255}
        else:
            base = dict()
            base['past_action'] = np.ravel(obs['action/environment'])
            for model_t in ['actor', 'critic']:
                if self.obs_preprocess.value == ObservationPreprocess.FLATTEN.value:
                    base[model_t] = flatten_dict(obs, self.flatten_order[model_t], model_t == 'actor')
                else:
                    base.update(actor_based_observation(obs, model_t == 'actor'))

        return base

    def reset(self):
        success_obs = None
        while not success_obs:
            try:
                success_obs = self.env.reset()
            except Exception as e:
                success_obs = None

        return self._observation(success_obs.observation)

    def render(self, mode="human"):
        cam = self.env.physics.render(camera_id='main_camera', width=480, height=240)
        cv2.imshow('Main Camera', cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def step(self, action: numpy.ndarray):
        _clone_action = action.copy()

        np_action = np.array([bins[action_index] for bins, action_index in zip(self.discrete_action_bin, action)])

        time_step = self.env.step(np_action)

        x = self._observation(time_step.observation)

        return x, time_step.reward, time_step.last(), {}

    def close(self):
        self.env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    import tabulate

    env = GymWrapper('rgb_train_random', ObservationPreprocess.ACTOR_BASED, 11, False)

    print(env.observation_space.spaces.keys())

    print('observation_space\n', tabulate.tabulate([[k, v.shape] for k, v in env.observation_space.spaces.items()]))
    print('action_space', env.action_space)

    reset_obs = env.reset()
    print(tabulate.tabulate([[k, v, v.shape] for k, v in reset_obs.items()]))

    action = env.action_space.sample()
    print('take random action', action)
    print(tabulate.tabulate([[k, v, v.shape] for k, v in env.step(action)[0].items()]))

    action = env.action_space.sample()
    print('take random action', action)
    print(tabulate.tabulate([[k, v, v.shape] for k, v in env.step(action)[0].items()]))

    env.close()


if __name__ == '__main__':
    from absl import app

    app.run(main)
