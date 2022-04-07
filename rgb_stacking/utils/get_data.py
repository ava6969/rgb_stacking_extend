import argparse
import pickle

import mpi4py as mp
import argparse as ap
import cv2
import tqdm, sys

from rgb_stacking.run import _mpi_init
from rgb_stacking.utils import environment, policy_loading
from dm_robotics.manipulation.props.rgb_objects import rgb_object

from rgb_stacking.utils.mpi_tools import proc_id, num_procs, gather

'''
TODO:
    1) SETUP MPI AND ARGPARSE
    2) RANDOMIZE VISUALS
    3) GATHER AND SERIALIZE DATA IN CHUNKS
'''

_POLICY_PATHS = lambda path: f'assets/saved_models/mpo_state_{path}'


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


def run(rank, test_triplet, total_frames: int, policy_path, debug=True, TOTAL_F=1E9):
    frames = []
    with environment.rgb_stacking(object_triplet=test_triplet,
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True) as env:

        policy = policy_loading.policy_from_path(policy_path)
        t_acquired = 0

        while t_acquired < total_frames:

            timestep = env.reset()
            state = policy.initial_state()
            frames.append(to_example(rank, policy_path.split('/')[0],
                                     test_triplet,timestep.observation, debug))
            t_acquired += 1
            done = False

            while not done:
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

    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Collection')
    parser.add_argument('-f', '--total_frames', type=int)
    parser.add_argument('-d', '--debug', action='store_true')
    triplet = lambda x: tuple(rgb_object.PROP_TRIPLETS_TEST.keys())[x]
    args = parser.parse_args()

    _mpi_init()
    rank = proc_id()
    sz = num_procs()

    frames_per_expert = args.total_frames // sz
    assert frames_per_expert > 0

    total_frames = run(rank, triplet(rank) if rank < len(rgb_object.PROP_TRIPLETS_TEST) else "rgb_train_random",
                       frames_per_expert,
                       _POLICY_PATHS( triplet(( rank // 5) ) ),
                       args.debug, args.total_frames)

    with open('data/rgb_example_{}.pkl'.format(rank), 'wb') as file:
        pickle.dump(total_frames, file)

