import argparse
import glob
# from torch._C import namedtuple
import sys
import time

import tqdm
from PIL import Image
from stable_baselines3.common.vec_env import CloudpickleWrapper
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from rgb_stacking.utils import mpi_tools
from rgb_stacking.utils.get_data import VisionModelGym

class Buffer:
    def __init__(self, rank, buffer_size, total_workers, no_dr, debug):

        self.env = VisionModelGym(rank, no_dr, debug )

        if rank == 0:
            print('Created Model/Resetting Env')
        self.env.reset()
        self.rank = rank

        if rank == 0:
            print('Reset successful')
        self.N = total_workers
        self.n = buffer_size // self.N

        if rank == 0:
            print("rank {}: Gathering {} Data Per {} workers".format(self.rank, self.n, self.N))

        self.img_size = [self.n, 200, 200, 3]
        self.pose_size = [self.n, 21]

        self.fl, self.fr, self.bl, self.poses = np.zeros(self.img_size, dtype=np.uint8), \
                                                np.zeros(self.img_size, dtype=np.uint8),\
                                                np.zeros(self.img_size, dtype=np.uint8), \
                                                np.zeros(self.pose_size, dtype=float)

    def gather(self):
        try:
            if self.rank == 0:
                for i in tqdm.tqdm(range(self.n)):
                    imgs, p = self.env.next()
                    self.fl[i], self.fr[i], self.bl[i], self.poses[i] = imgs['fl'], imgs['fl'], imgs['fl'], p
            else:
                for i in range(self.n):
                    imgs, p = self.env.next()
                    self.fl[i], self.fr[i], self.bl[i], self.poses[i] = imgs['fl'], imgs['fl'], imgs['fl'], p

        except Exception as e:
            print(e)
            self.env.close()
            sys.exit()


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "gather":
                env.gather()
                remote.send((env.fr, env.fl, env.bl, env.poses))
            elif cmd == "close":
                env.env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class VecBuffer:
    def __init__(self, buffer_size, total_workers, no_dr, debug, start_method=None):
        self.waiting = False
        self.closed = False

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        self.ctx = mp.get_context(start_method)
        self.job_queue = self.ctx.Queue(maxsize=64)

        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(total_workers)])
        self.processes = []

        def make_env(rank, buffer_size, total_workers, no_dr, debug):
            def thunk():
                return Buffer(rank, buffer_size, total_workers, no_dr, debug)
            return thunk

        fns = [ make_env(i, buffer_size, total_workers, no_dr, debug) for i in range(total_workers) ]
        for work_remote, remote, env_fn in tqdm.tqdm( zip(self.work_remotes, self.remotes, fns) ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = self.ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset" , None))

        for remote in self.remotes:
            remote.recv()

    def gather(self, N_total_batches):
        try:
            i = 0

            for i in range(N_total_batches):
                for remote in self.remotes:
                    remote.send(("gather", None))
                _data = [remote.recv() for remote in self.remotes]
                fr, fl, bl, poses = zip(*_data)
                yield np.vstack(fr), np.vstack(fl), np.vstack(bl), np.vstack(poses)
        except Exception as e:
            print(e)
            sys.exit()

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()


class CustomDataset(Dataset):
    def __init__(self, train_batch, img_transform=None, target_transform=None):

        self.train_batch = train_batch
        self.transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_batch['fl'])

    @staticmethod
    def norm(x):
        # x[:3] = x[:3] / 0.25
        return x

    def __getitem__(self, idx):
        images = { k : self.transform( torch.from_numpy(
                                        self.train_batch[k][idx].astype(float) ).permute(2, 0, 1).float() )
                   for k in ['fl', 'fr', 'bl'] }

        pose = self.train_batch['poses'][idx]
        pose[:7], pose[7:14], pose[14:] = self.norm(pose[:7]), self.norm(pose[7:14]), self.norm(pose[14:])

        label = torch.from_numpy( pose ).float()

        return images, label

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def _load_data(parent_path, dfs):
    batch = []
    for df_file in tqdm.tqdm(dfs):
        df = pd.read_csv(df_file)
        rank = df_file.split('/')[-1][:-4].split('_')[-1]
        for i, id in enumerate(df['id']):
            image_path = [f'{parent_path}/data/images/IMG_{pov}_{id}_{rank}.png' for pov in ['fl', 'fr', 'bl']]
            batch.append( (image_path, np.array([float(df[k][i]) for k in KEYS], float) ) )
    return batch

def load_data(parent_path, sz=None, jobs=1):
    dfs = glob.glob(parent_path + '/data/*csv')
    dfs = dfs if sz is None else dfs[:sz]
    
    if sz is not None:
        jobs= min(jobs, sz)
    
    assert jobs > 0
    delta = max(1, len(dfs) // jobs)
    args = [ (parent_path, dfs[i:i+delta]) for i in range(0, len(dfs), delta)]

    if jobs > 1:
        with mp.Pool() as pool:
            batches = pool.starmap(_load_data, args)
            batch = []
            for b in batches:
                batch.extend(b)
            return batch
    else:
        return _load_data(*args[0])

def view(batch, label):
    fl, fr, bl = batch

    fl = Image.fromarray( fl.cpu().to(torch.uint8).numpy() )
    fr = Image.fromarray( fr.cpu().to(torch.uint8).numpy())
    bl = Image.fromarray( bl.cpu().to(torch.uint8).numpy() )

    fl.show('fl')
    fr.show('fr')
    bl.show('bl')

    print(','.join( f'{k}={label[i]}' for i, k in enumerate(KEYS)))
    return fl, fr, bl