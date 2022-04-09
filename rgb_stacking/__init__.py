from rgb_stacking.contrib.gym_wrapper import ObservationPreprocess
import mpi4py

ROOT = 0
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True

ACTION_BIN_sIZE = 11

