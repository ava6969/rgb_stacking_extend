import glob
# from torch._C import namedtuple
import sys
import time

import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from rgb_stacking.utils import mpi_tools
from rgb_stacking.utils.get_data import VisionModelGym


KEYS = ['rX', 'rY', 'rZ', 'rQ1', 'rQ2', 'rQ3', 'rQ4',
        'bX', 'bY', 'bZ', 'bQ1', 'bQ2', 'bQ3', 'bQ4',
        'gX', 'gY', 'gZ', 'gQ1', 'gQ2', 'gQ3', 'gQ4']


class Buffer:
    def __init__(self, buffer_size, total_workers, no_dr, debug):
        time.sleep(5)
        self.env = VisionModelGym( mpi_tools.proc_id(), no_dr, debug )
        print('Created Model/Resetting Env')
        self.env.reset()
        print('Reset successful')
        self.N = total_workers
        self.n = buffer_size // self.N
        print("Gathering {} Data Per {} workers".format(self.n, self.N))

        self.img_size = [self.n, 200, 200, 3]
        self.pose_size = [self.n, 21]

        self.fl, self.fr, self.bl, self.poses = np.empty(self.img_size, dtype=np.uint8), \
                                                np.empty(self.img_size, dtype=np.uint8),\
                                                np.empty(self.img_size, dtype=np.uint8), \
                                                np.empty(self.pose_size, dtype=float)

    def gather(self):
        try:
            for i in tqdm.tqdm(range(self.n)):
                imgs, p = self.env.next()
                self.fl[i], self.fr[i], self.bl[i], self.poses[i] = imgs['fl'], imgs['fl'], imgs['fl'], p
        except Exception as e:
            print(e)
            self.env.close()
            sys.exit()


class CustomDataset(Dataset):
    def __init__(self, train_batch, img_transform=None, target_transform=None):

        self.train_batch = train_batch
        self.transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_batch['fl'])

    @staticmethod
    def norm(x):
        return (x + 0.25) / 0.50

    def __getitem__(self, idx):
        images = { k : self.transform( torch.from_numpy(
                                        self.train_batch[k][idx].astype(float) ).permute(2, 0, 1).float() )
                   for k in ['fl', 'fr', 'bl'] }

        pose = self.train_batch['poses'][idx]
        pose[:3], pose[7:10], pose[14 : 17] = self.norm(pose[:3]), self.norm(pose[7:10]), self.norm(pose[14:17])

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