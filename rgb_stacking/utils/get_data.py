import time

import rgb_stacking
from rgb_stacking.utils.dr.gym_dr import VisionModelDomainRandomizer

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

from rgb_stacking.utils.mpi_tools import proc_id, num_procs, msg
import colorsys
import pandas as pd
from rgb_stacking.utils.caliberate import relative_pose
from dm_control import viewer

flags.DEFINE_integer('total_frames', 10000, 'path to root folder of dataset')
flags.DEFINE_integer('rank', None, 'path to root folder of dataset')
flags.DEFINE_integer('split', None, 'path to root folder of dataset')

'''
TODO:
    2) RANDOMIZE VISUALS
'''

KEYS = ['rX', 'rY', 'rZ', 'rQ1', 'rQ2', 'rQ3', 'rQ4',
        'bX', 'bY', 'bZ', 'bQ1', 'bQ2', 'bQ3', 'bQ4',
        'gX', 'gY', 'gZ', 'gQ1', 'gQ2', 'gQ3', 'gQ4']


def to_example(rank, policy, env, obs, _debug, only_plot=False):
    images = {'bl': obs['basket_back_left/pixels'],
              'fl': obs['basket_front_left/pixels'],
              'fr': obs['basket_front_right/pixels']}

    if _debug:
        for k in ['bl', 'fl', 'fr']:
            cv2.imshow('{}:{}:P{}:E{}'.format(k, rank, policy, env), images[k])
        cv2.waitKey(10)

    if not only_plot:
        poses = list(relative_pose(obs['rgb30_red/abs_pose'][-1])) + \
                list(relative_pose(obs['rgb30_blue/abs_pose'][-1])) + \
                list(relative_pose(obs['rgb30_green/abs_pose'][-1]))

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


class VisionModelGym:

    def __init__(self, rank, no_dr, debug):

        POLICY_PATHS = lambda path: f'rgb_stacking/utils/assets/saved_models/mpo_state_{path}'
        triplet = lambda x: tuple(rgb_object.PROP_TRIPLETS_TEST.keys())[x]

        self.test_triplet = triplet(rank % 5)
        self.policy_path = POLICY_PATHS(self.test_triplet)
        self.rank = rank
        self.debug = debug

        self.env = environment.rgb_stacking(object_triplet=self.test_triplet,
                                            observation_set=environment.ObservationSet.ALL,
                                            use_sparse_reward=True, frameStack=3)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices([], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.list_logical_devices('GPU')

                if rank == 0:
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU, using:", gpus[0])
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        with tf.device('/cpu:0'):
            self.policy = policy_loading.policy_from_path(self.policy_path)

        if no_dr:
            self.randomize = None
        else:
            self.randomize = VisionModelDomainRandomizer(self.env)

        self.env_state = None
        self.policy_state = None

    def reset(self):
        self.env_state = None
        while not self.env_state:
            try:
                self.env_state = self.env.reset()
            except Exception as e:
                print(e)
                self.env_state = None

        with tf.device('/cpu:0'):
            self.policy_state = self.policy.initial_state()

    def next(self):

        if self.randomize is not None:
            self.randomize()

        with tf.device('/cpu:0'):
            (action, _), self.policy_state = self.policy.step(self.env_state, self.policy_state)

        self.env_state = self.env.step(action)

        if (self.env_state.last()) or (self.env_state.reward == 1):
            self.reset()

        return to_example(self.rank, self.policy_path.split('/')[0],
                          self.test_triplet, self.env_state.observation, self.debug)

    def close(self):
        self.env.close()


def run(rank, total_frames: int, debug=True, TOTAL_F=1E9):
    frames = []
    try:
        data_handler = VisionModelGym(rank, debug, True)
        t_acquired = 0
        last = time.time()
        data_handler.reset()

        while t_acquired < total_frames:

            t = 0
            while t_acquired < total_frames:
                frames.append(data_handler.next())

                t_acquired += 1
                if t_acquired % 100 == 0:
                    total = t_acquired * num_procs()
                    if proc_id() == 0:
                        c = time.time()
                        elapsed = c - last
                        _str = f'Total Frames Acquired: [ {total}/{TOTAL_F}] frames, FPS: {total / elapsed}'
                        msg(_str)
                t += 1
    except Exception as e:
        print(e)
        data_handler.close()

    return frames


def main(_argv):
    parser = argparse.ArgumentParser('Data Collection')
    parser.add_argument('-f', '--total_frames', type=int)
    parser.add_argument('-l', '--debug_specs', type=bool, default=False)
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-r', '--rank', type=int)
    parser.add_argument('-s', '--split', type=int)

    args = parser.parse_args()
    j = 0

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
        total_frames = run(rank,
                           frames_per_expert,
                           args.debug_specs, frames_per_expert * sz)

        _dict = defaultdict(lambda: list())
        for img, pose in total_frames:
            cv2.imwrite('rgb_stacking/data/images/IMG_bl_{}_{}.png'.format(j, rank), img['bl'])
            cv2.imwrite('rgb_stacking/data/images/IMG_fl_{}_{}.png'.format(j, rank), img['fl'])
            cv2.imwrite('rgb_stacking/data/images/IMG_fr_{}_{}.png'.format(j, rank), img['fr'])
            for i_k, k in enumerate(KEYS):
                _dict[k].append(pose[i_k])
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
