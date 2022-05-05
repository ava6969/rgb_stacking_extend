from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gym

env = VecFrameStack( DummyVecEnv([ lambda : gym.make("PongNoFrameskip-v4") ]), 4, 'first')