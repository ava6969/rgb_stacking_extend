import numpy as np
from gym.utils import seeding


class NoiseDistribution:
    def __init__(self, name, seed=1):
        self.name = name
        self._np_random = None
        self._np_random, self.seed = seeding.np_random(seed)

    def sample(self):
        raise NotImplementedError


class Normal(NoiseDistribution):
    def __init__(self, loc, scale, seed=1):
        super().__init__('normal', seed)
        self.loc = loc
        self.scale = scale

    def sample(self):
        return self._np_random.normal(self.loc, self.scale)


class Uniform(NoiseDistribution):
    def __init__(self, low, hi, seed=1):
        super().__init__('normal', seed)
        self.low = low
        self.hi = hi

    def sample(self):
        return self._np_random.uniform(self.low, self.hi)


class LogUniform(NoiseDistribution):
    def __init__(self, low, hi, seed=1):
        super().__init__('normal', seed)
        self.low = low
        self.hi = hi

    def sample(self):
        return np.exp(self._np_random.uniform(self.low, self.hi))


if __name__ == '__main__':
    rv = Uniform([0.3, 0.1, 0.05], [0.6, 0.1, 0.005])
    print(rv.sample())