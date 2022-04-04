import dataclasses
from noise import NoiseDistribution, Normal, LogUniform, Uniform
import numpy as np


class DomainRandomizer:
    def __init__(self, sim):
        self.sim = sim

        self.params = dict(
            gripper_friction=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            hand_friction=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            arm_friction=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            basket_friction=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            object_friction=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            object_mass=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
            arm_joint_armature=Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005]),
        )

    def reset(self):
        for key, param in zip(self.params):
            self.mod(key, param.sample())
        self.update_sim()

    def mod(self, key, val):
        name, attr = key.split('_')
        getattr(self, f"mod_{attr}")(name, val)

    def mod_friction(self, name, val):
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_friction[geom_id] = np.array(val)

    def update_sim(self):
        self.sim.forward()