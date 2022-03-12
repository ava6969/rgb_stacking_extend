#!/bin/bash -l
# NOTE the -l flag!
#SBATCH -J PPO_ATTN_ACTOR_BASED
#SBATCH --output %x_%j.out
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

#SBATCH --gres=gpu:p4:1
export OPENBLAS_NUM_THREADS=1
conda activate shapes
cd /home/ava6969/rgb_stacking_extend
time  MUJOCO_GL=egl python3 rgb_stacking/run.py --config_path rgb_stacking/contrib/configs/PPO_ATTN_ACTOR_BASED.yaml
