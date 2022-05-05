
from rgb_stacking.utils.dr.noise import Uniform
import numpy as np


class VisionModelDomainRandomizer:
    
    def __init__(self, env):
        self.env = env
        self.props_color_geom = [p.mjcf_model.find_all('geom')[0] for p in self.env.base_env.task.props]

        self.light = self.env.base_env.task.root_entity.mjcf_model.find_all('light')[0]

        self.ambient = self.get_range_single(0.3, 3, 0.1)
        self.diffuse = self.get_range_single(0.6, 3, 0.1)

        self.camera = self.env.base_env.task.root_entity.mjcf_model.find_all('camera')[1:4]

        self.camera_left_pos = self.get_range([1, -0.395, 0.253], 0.1)
        # camera_left_euler = get_range([1.142, 0.004, 0.783], 0.05)
        self.camera_right_pos = self.get_range([0.967, 0.381, 0.261], 0.1)

        self.camera_back = self.get_range([0.06, -0.26, 0.39], 0.1)
        # camera_right_euler = get_range([1.088, 0.001, 2.362], 0.05)
        self.camera_fov = Uniform(35, 45)
        
    @staticmethod
    def get_range(x, pct):
        lo = [x_* (1-pct) for x_ in x]
        hi = [x_ * (1 + pct) for x_ in x]
        return Uniform(lo, hi)

    @staticmethod
    def get_range_single(x, sz, pct):
        x = np.full(sz, x)
        lo = [x_* (1-pct) for x_ in x]
        hi = [x_ * (1 + pct) for x_ in x]
        return Uniform(lo, hi)
        
    
    def __call__(self, ):
        _light = self.env.physics.bind(self.light)
        _light.ambient =  self.ambient.sample()
        _light.diffuse = self.diffuse.sample()

        _cam = self.env.physics.bind(self.camera[0])
        _cam.fovy = self.camera_fov.sample()

        _cam = self.env.physics.bind(self.camera[1])
        _cam.fovy = self.camera_fov.sample()

        _cam = self.env.physics.bind(self.camera[2])
        _cam.fovy = self.camera_fov.sample()
        