import pickle, glob

import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from rgb_stacking.utils.get_data import KEYS
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, examples, img_transform=None, target_transform=None):
        self.examples = examples
        self.transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_paths, pose = self.examples[idx]
        images = torch.stack( [ self.transform( torch.from_numpy(
            np.array( Image.open( path ) ).astype(float) ).permute(2, 0, 1).float() ) for path in img_paths ] )
        label = torch.from_numpy( pose ).float()
        return images, label

def load_data(parent_path, sz=None):
    dfs = glob.glob(parent_path + '/data/*csv')
    dfs = dfs if sz is None else dfs[:sz]
    batch = []

    for df_file in tqdm.tqdm(dfs[:sz]):
        df = pd.read_csv(df_file)
        rank = df_file.split('/')[-1][:-4].split('_')[-1]
        for i, id in enumerate(df['id']):
            image_path = [f'{parent_path}/data/images/IMG_{pov}_{id}_{rank}.png' for pov in ['fl', 'fr', 'bl']]
            batch.append( (image_path, np.array([float(df[k][i]) for k in KEYS], float) ) )
    return batch

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