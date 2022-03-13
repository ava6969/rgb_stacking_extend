#!/bin/bash -l
# NOTE the -l flag!
#SBATCH -J PPO_ATTN_LRe3
#SBATCH --error %x_%j.err
#SBATCH --mail-user=ava6969@rit.edu
#SBATCH --mail-type=ALL

#SBATCH -t 2-10:00:0

#Put the job in the appropriate partition matching the account and request FOUR cours

#SBATCH --account shapes
#SBATCH --partition tier3
#SBATCH --ntasks 1
#SBATCH -c 36

#Job membory requirements in MB=m (default), GB=g, or TB=t
#SBATCH --mem=300g

#SBATCH --gres=gpu:v100:2
export OPENBLAS_NUM_THREADS=1
conda activate shapes
cd /home/ava6969/rgb_stacking_extend
time  MUJOCO_GL=egl python3 rgb_stacking/run.py --config_path rgb_stacking/contrib/configs/PPO_ATTN_LRe3.yaml
