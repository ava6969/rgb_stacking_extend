import warnings
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import cv2
from dm_control import viewer
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.moma import subtask_env
import numpy as np
from rgb_stacking.utils.task import _BASKET_ORIGIN, WORKSPACE_CENTER, WORKSPACE_SIZE

from rgb_stacking.utils import environment, policy_loading

warnings.filterwarnings('ignore')
_TEST_OBJECT_TRIPLETS = tuple(rgb_object.PROP_TRIPLETS_TEST.keys())

_VALID_SIZE = np.array([0.5, 0.5, 0.5])
min_workspace_limits = - _VALID_SIZE / 2
max_workspace_limits =  _VALID_SIZE / 2


def relative_pose(pose):
    pose[:3] = np.clip(pose[:3] - WORKSPACE_CENTER, min_workspace_limits, max_workspace_limits)
    return pose


def calibrate(argv: Sequence[str]):
    policy = policy_loading.policy_from_path('rgb_stacking/utils/assets/saved_models/mpo_state_rgb_test_triplet1')
    with environment.rgb_stacking(object_triplet=tuple(rgb_object.PROP_TRIPLETS_TEST.keys())[0],
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True) as env:

        camera = env.base_env.task.root_entity.mjcf_model.find_all('camera')[1:4]
        objects = [p.mjcf_model.find_all('geom')[0] for p in env.base_env.task.props]
        action_spec = env.action_spec()

        global state, _BASKET_ORIGIN, WORKSPACE_CENTER

        state = policy.initial_state()

        def _policy(timestep):
            global state
            obs = timestep.observation

            for k in ['basket_back_left/pixels', 'basket_front_left/pixels', 'basket_front_right/pixels']:
                cv2.imshow('{}'.format(k), obs[k])
            cv2.waitKey(1)

            print(WORKSPACE_CENTER, min_workspace_limits, max_workspace_limits)
            r, g, b = obs['rgb30_red/abs_pose'][-1], obs['rgb30_green/abs_pose'][-1], obs['rgb30_blue/abs_pose'][-1]

            print(f'r: {relative_pose(r)}\n'
                  f'g: {relative_pose(g)}\n'
                  f'b: {relative_pose(b)}' )

            (action, _), state = policy.step(timestep, state)

            return action

        viewer.launch(env, _policy)


if __name__ == '__main__':
  app.run(calibrate)
