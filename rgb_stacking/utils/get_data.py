import time

import rgb_stacking
rgb_stacking.LOAD_GYM = False
import argparse
import os
from collections import defaultdict
from absl import flags, app
from absl.flags import FLAGS
import mpi4py
import mpi4py as mp
import cv2
import tensorflow as tf
from rgb_stacking.utils import environment, policy_loading
from dm_robotics.manipulation.props.rgb_objects import rgb_object
import numpy as np
from rgb_stacking.utils.dr.noise import Uniform
from rgb_stacking.utils.mpi_tools import proc_id, num_procs, msg
import colorsys
import pandas as pd


flags.DEFINE_integer('total_frames', 10000, 'path to root folder of dataset')
flags.DEFINE_integer('rank', None,'path to root folder of dataset')
flags.DEFINE_integer('split', None,'path to root folder of dataset')

'''
TODO:
    2) RANDOMIZE VISUALS
'''

_POLICY_PATHS = lambda path: f'rgb_stacking/utils/assets/saved_models/mpo_state_{path}'

KEYS = ['rX', 'rY', 'rZ', 'rQ1', 'rQ2', 'rQ3', 'rQ4',
        'bX', 'bY', 'bZ', 'bQ1', 'bQ2', 'bQ3', 'bQ4',
        'gX', 'gY', 'gZ', 'gQ1', 'gQ2', 'gQ3', 'gQ4']


def to_example(rank, policy, env, obs, _debug):
    images = {'bl': obs['basket_back_left/pixels'],
              'fl': obs['basket_front_left/pixels'],
              'fr': obs['basket_front_right/pixels']}
    poses = list(obs['rgb30_red/abs_pose'][-1]) + \
            list(obs['rgb30_blue/abs_pose'][-1]) + \
            list(obs['rgb30_green/abs_pose'][-1])

    if _debug:
        for k in ['bl', 'fl', 'fr']:
            cv2.imshow( '{}:{}:P{}:E{}'.format(k, rank, policy, env), images[k])
        cv2.waitKey(10)

    return images, poses


def init_env():
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        IN_MPI="1"
    )
    return env


def _mpi_init():
    mpi4py.MPI.Init()
    msg('Successfully loaded')


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

        camera_back = get_range([0.06, -0.26, 0.39], 0.1)
        # camera_right_euler = get_range([1.088, 0.001, 2.362], 0.05)
        camera_fov = Uniform(30, 40)

        r = Uniform([0, 0.5, 0.5], [70/360, 1, 1])
        g = Uniform([95/360, 0.5, 0.5 ], [165/360, 1 , 1 ])
        b = Uniform([200/255, 0.5 , 0.5 ], [270/360, 1 , 1 ])

        t_acquired = 0
        sampler = Uniform([mass, mass, mass, mass, mass, mass], [1, 0, 1, 1.0, 0.0, 1.0])
        last = time.time()
        while t_acquired < total_frames:
            timestep = None
            while not timestep:
                try:
                    timestep = env.reset()
                except Exception as e:
                    timestep = None

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

                _cam = env.physics.bind(camera[2])
                _cam.pos = camera_back.sample()
                _cam.fovy = camera_fov.sample()

                if np.random.rand() > 0.5:
                    x = 0.99 ** t * force
                    env.physics.bind(props[0]).xfrc_applied[:2] = np.random.normal(x[:2], mass).clip(min=0)
                    env.physics.bind(props[1]).xfrc_applied[:2] = np.random.normal(x[:2], mass).clip(min=0)
                    env.physics.bind(props[2]).xfrc_applied[:2] = np.random.normal(x[:2], mass).clip(min=0)

                (action, _), state = policy.step(timestep, state)
                timestep = env.step(action)
                frames.append(to_example(rank, policy_path.split('/')[0],
                                         test_triplet, timestep.observation, debug))
                done = timestep.last() or (timestep.reward == 1)

                t_acquired += 1
                if t_acquired % 100 == 0:
                    total = mp.MPI.COMM_WORLD.allreduce(t_acquired)
                    if proc_id() == 0:
                        c =  time.time()
                        elapsed = c - last
                        _str = f'Total Frames Acquired: [ {total}/{TOTAL_F}] frames, FPS: {total/elapsed}'
                        msg(_str)
                t += 1
        del policy

    return frames


def main(_argv):
    parser = argparse.ArgumentParser('Data Collection')
    parser.add_argument('-f', '--total_frames', type=int)
    parser.add_argument('-l', '--debug_specs', type=bool, default=False)
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-r', '--rank', type=int)
    parser.add_argument('-s', '--split', type=int)
    triplet = lambda x: tuple(rgb_object.PROP_TRIPLETS_TEST.keys())[x]
    args = parser.parse_args()
    j = 0

    init_env()
   
    _mpi_init()
    rank = proc_id() if args.rank is None else args.rank
    sz = num_procs()
    print("Rank: ", rank)

    if rank == 0:
        if not os.path.exists('rgb_stacking/data'):
            os.mkdir('rgb_stacking/data')
            os.mkdir('rgb_stacking/data/images')

    split = args.split
    frames_per_expert = args.total_frames // sz // split
    assert frames_per_expert > 0


    for i in range(split):
        # Run inference on CPU
        with tf.device('/cpu'):
            total_frames = run(rank, triplet(rank) if rank < len(rgb_object.PROP_TRIPLETS_TEST) else "rgb_train_random",
                               frames_per_expert,
                               _POLICY_PATHS( triplet( rank % 5 ) ),
                               args.debug_specs, frames_per_expert*sz)

        _dict = defaultdict(lambda: list())
        for img, pose in total_frames:
            cv2.imwrite('rgb_stacking/data/images/IMG_bl_{}_{}.png'.format(j, rank), img['bl'])
            cv2.imwrite('rgb_stacking/data/images/IMG_fl_{}_{}.png'.format(j, rank), img['fl'])
            cv2.imwrite('rgb_stacking/data/images/IMG_fr_{}_{}.png'.format(j, rank), img['fr'])
            for i_k, k in enumerate(KEYS):
                _dict[k].append( pose[i_k] )
            _dict['id'].append(j)
            j += 1
        pd.DataFrame(_dict).to_csv(f'rgb_stacking/data/data_batch_{i}_{rank}.csv')
        msg('saved batch {}'.format(i))
        del total_frames


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass