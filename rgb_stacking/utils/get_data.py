from rgb_stacking.utils import environment


def get():
    env = environment.rgb_stacking(object_triplet=ALL,
                                   observation_set=environment.ObservationSet.ALL)
