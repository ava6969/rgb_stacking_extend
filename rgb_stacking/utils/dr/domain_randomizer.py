import dataclasses

from dm_control import mjcf

from rgb_stacking.utils.dr.noise import NoiseDistribution, Normal, LogUniform, Uniform
import numpy as np
from dm_robotics.moma.entity_initializer import base_initializer


@dataclasses.dataclass
class DomainRandomizer(base_initializer.Initializer):
    def __init__(self, basket, props, robot):
        self.props = props
        self.basket = basket
        self.arm = robot.arm
        self.gripper = robot.gripper

        friction, mass, low, hi = np.array([0.9, 0.001, 0.001], float), 0.201, 0.9, 1.1
        self.object_rand = dict(friction=Uniform(friction * low, friction * hi),
                                mass=Uniform(mass * low, mass * hi))

        friction = np.array([0.1, 0.1, 0.0001], float)
        self.arm_rand = dict(friction=Uniform(friction * low, friction * hi),
                             damping=Uniform(0.1 * low, 0.1 * hi),
                             armature=Uniform(low, hi),
                             friction_loss=Uniform(0.3 * low, 0.3 * hi))

        friction = np.array([1, 0.005, 0.0001], float)
        self.hand_rand = dict(friction=Uniform(friction * low, friction * hi),
                              driver_damping=Uniform(0.1 * low, 0.1 * hi),
                              armature=Uniform(low, hi),
                              spring_link_damping=Uniform(0.3 * low, 0.3 * hi))

        friction = np.array([1.0, 0.001, 0.001], float)
        self.basket_friction = Uniform(friction * low, friction * hi)

        friction = np.array([1, 0, 0, 0, 0, 0], float)
        self.actuator_gear = Uniform(friction * low, friction * hi)

    def __call__(self, physics: mjcf.Physics, random_state: np.random.RandomState) -> bool:
        for p in self.props:
            collision_geom = p.mjcf_model.find_all('geom')[1]
            collision_geom.friction = self.object_rand['friction'].sample()

        basket_geoms = self.basket.mjcf_model.find_all('geom')
        for b in basket_geoms:
            b.friction = self.basket_friction.sample()

        hand_driver = self.gripper.mjcf_model.find('default', 'driver')
        hand_spring_link = self.gripper.mjcf_model.find('default', 'spring_link')
        hand = self.gripper.mjcf_model.find('default', 'reinforced_fingertip')

        hand.geom.friction = self.hand_rand['friction'].sample()
        hand_driver.joint.armature = self.hand_rand['armature'].sample()
        hand_driver.joint.damping = self.hand_rand['driver_damping'].sample()
        hand_spring_link.joint.damping = self.hand_rand['spring_link_damping'].sample()

        for joint in self.arm.joints:
            joint.armature = self.arm_rand['armature'].sample()
            joint.damping = self.arm_rand['damping'].sample()
            joint.frictionloss = self.arm_rand['friction_loss'].sample()

        for actuator in self.arm.actuators:
            actuator.gear = self.actuator_gear.sample()

        geoms = self.arm.mjcf_model.find_all('geom')
        for g in geoms:
            g.friction = self.arm_rand['friction'].sample()

        return True
