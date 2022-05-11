import mpi4py
import torchvision.transforms
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from rgb_stacking.utils.pose_estimator.dataset import VecBuffer
import torch

transform = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])

def stat(df, name):
 df = transform(torch.from_numpy(fr).float().permute(0, 3, 1, 2))
 image_size = 200
 psum = torch.tensor([0.0, 0.0, 0.0])
 psum_sq = torch.tensor([0.0, 0.0, 0.0])
 psum += df.sum(axis=[0, 2, 3])
 psum_sq += (df ** 2).sum(axis=[0, 2, 3])
 count = len(df) * image_size * image_size
 total_mean = psum / count
 total_var = (psum_sq / count) - (total_mean ** 2)
 total_std = torch.sqrt(total_var)

 print(name, ":", total_mean, total_std)

if __name__ == '__main__':
 data_gen = VecBuffer(1000000, 72, True, False, "forkserver")
 for data in data_gen.gather(1):
  fr, fl, bl, _ = data
  stat(fr, 'fr')
  stat(fl, 'bl')
  stat(bl, 'bl')
 data_gen.close()
