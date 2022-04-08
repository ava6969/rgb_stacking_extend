import argparse
import pickle

import mpi4py as mp
import argparse as ap
import cv2
import tqdm, sys
import tensorflow as tf
from rgb_stacking.run import _mpi_init, init_env
from rgb_stacking.utils import environment, policy_loading
from dm_robotics.manipulation.props.rgb_objects import rgb_object
import numpy as np
from rgb_stacking.utils.dr.noise import Uniform, LogUniform
from rgb_stacking.utils.mpi_tools import proc_id, num_procs, gather, msg
import colorsys
from scipy.spatial.transform import Rotation as R

'''
TODO:
    2) RANDOMIZE VISUALS
'''

_POLICY_PATHS = lambda path: f'rgb_stacking/utils/assets/saved_models/mpo_state_{path}'


def to_example(rank, policy, env, obs, _debug):
    images = {'bl': obs['basket_back_left/pixels'],
              'fl': obs['basket_front_left/pixels'],
              'fr': obs['basket_front_right/pixels']}
    poses = {'r': obs['rgb30_red/abs_pose'],
             'b': obs['rgb30_blue/abs_pose'],
             'g': obs['rgb30_green/abs_pose']}

    if _debug:
        for k in ['bl', 'fl', 'fr']:
            cv2.imshow( '{}:{}:P{}:E{}'.format(k, rank, policy, env), images[k])
        cv2.waitKey(1)

    return images, poses


def get_range(x, pct):
    lo = [x_* (1-pct) for x_ in x]
    hi = [x_ * (1 + pct) for x_ in x]
    return Uniform(lo, hi)

def get_range_single(x, sz, pct):
    x = np.full(sz, x)
    lo = [x_* (1-pct) for x_ in x]
    hi = [x_ * (1 + pct) for x_ in x]
    return Uniform(lo, hi)

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]

def run(rank, test_triplet, total_frames: int, policy_path, debug=True, TOTAL_F=1E9):

    frames = []
    with environment.rgb_stacking(object_triplet=test_triplet,
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True) as env:

        policy = policy_loading.policy_from_path(policy_path)

        props = [p.mjcf_model.find_all('body')[2] for p in env.base_env.task.props]
        props_color_geom = [p.mjcf_model.find_all('geom')[0] for p in env.base_env.task.props]
        mass = env.physics.bind(props[1]).mass

        light = env.base_env.task.root_entity.mjcf_model.find_all('light')[0]

        ambient = get_range_single(0.3, 3, 0.1)
        diffuse = get_range_single(0.6, 3, 0.1)

        camera = env.base_env.task.root_entity.mjcf_model.find_all('camera')[1:4]
        camera_left_pos = get_range([1, -0.395, 0.253], 0.1)
        # camera_left_euler = get_range([1.142, 0.004, 0.783], 0.05)
        camera_right_pos = get_range([0.967, 0.381, 0.261], 0.1)
        # camera_right_euler = get_range([1.088, 0.001, 2.362], 0.05)
        camera_fov = Uniform(30, 40)

        r = Uniform([0, 0.5, 0.5], [70/360, 1, 1])
        g = Uniform([95/360, 0.5, 0.5 ], [165/360, 1 , 1 ])
        b = Uniform([200/255, 0.5 , 0.5 ], [270/360, 1 , 1 ])

        t_acquired = 0
        sampler = Uniform([0.0, 0.0, 0.0, 0, 0, 0], [1, 1, 0.3, 0, 0, 0])
        while t_acquired < total_frames:

            timestep = env.reset()
            state = policy.initial_state()
            frames.append(to_example(rank, policy_path.split('/')[0],
                                     test_triplet,timestep.observation, debug))
            t_acquired += 1
            done = False
            force = sampler.sample()
            t = 0

            while not done and t_acquired < total_frames:
                env.physics.bind(props_color_geom[0]).rgba[:3] = colorsys.hsv_to_rgb(*r.sample())
                env.physics.bind(props_color_geom[1]).rgba[:3] = colorsys.hsv_to_rgb(*g.sample())
                env.physics.bind(props_color_geom[2]).rgba[:3] = colorsys.hsv_to_rgb(*b.sample())

                _light = env.physics.bind(light)
                _light.ambient =  ambient.sample()
                _light.diffuse = diffuse.sample()

                _cam = env.physics.bind(camera[0])
                _cam.pos = camera_left_pos.sample()
                _cam.fovy = camera_fov.sample()
                # _cam.quat = get_quaternion_from_euler( *camera_left_euler.sample() )

                _cam = env.physics.bind(camera[1])
                _cam.pos = camera_right_pos.sample()
                _cam.fovy = camera_fov.sample()
                # _cam.quat = get_quaternion_from_euler( *camera_right_euler.sample() )

                if np.random.rand() > 0.5:
                    x = 0.99 ** t * force[:3]
                    env.physics.bind(props[0]).xfrc_applied[:3] = np.random.normal(x, mass).clip(min=0)
                    env.physics.bind(props[1]).xfrc_applied[:3] = np.random.normal(x, mass).clip(min=0)
                    env.physics.bind(props[2]).xfrc_applied[:3] = np.random.normal(x, mass).clip(min=0)

                (action, _), state = policy.step(timestep, state)
                timestep = env.step(action)
                frames.append(to_example(rank, policy_path.split('/')[0],
                                         test_triplet, timestep.observation, debug))
                done = timestep.last() or (timestep.reward == 1)

                t_acquired += 1
                if t_acquired % 100 == 0:
                    total = mp.MPI.COMM_WORLD.allreduce(t_acquired)
                    if proc_id() == 0:
                        print('Total Frames Acquired: [', total, f'/{TOTAL_F}] frames')
                t += 1



    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Collection')
    parser.add_argument('-f', '--total_frames', type=int)
    parser.add_argument('-d', '--debug', action='store_true')
    triplet = lambda x: tuple(rgb_object.PROP_TRIPLETS_TEST.keys())[x]
    args = parser.parse_args()

    env = init_env()
    env['CUDA_VISIBLE_DEVICES']=0
    _mpi_init()
    rank = proc_id()
    sz = num_procs()

    frames_per_expert = args.total_frames // sz
    assert frames_per_expert > 0

    # Run inference on CPU
    with tf.device('/cpu:0'):
        total_frames = run(rank, triplet(rank) if rank < len(rgb_object.PROP_TRIPLETS_TEST) else "rgb_train_random",
                           frames_per_expert,
                           _POLICY_PATHS( triplet( rank // 5 ) ),
                           args.debug, args.total_frames)

    with open('rgb_stacking/data/rgb_example_{}.pkl'.format(rank), 'wb') as file:
        pickle.dump(total_frames, file)

