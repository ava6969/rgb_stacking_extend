import gym

gym.envs.register(
     id='StackRGBTestTriplet-v0',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_random', 'flatten': True, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTriplet-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1', 'flatten': True, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTriplet-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2', 'flatten': True, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTriplet-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3', 'flatten': True, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTriplet-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4', 'flatten': True, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTriplet-v5',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5', 'flatten': True, 'discrete_n': 21}
)


gym.envs.register(
     id='StackRGBTestTripletDict-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v5',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTrain-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'flatten': True, 'discrete_n': None}
)

gym.envs.register(
     id='StackRGBTrain-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'flatten': False, 'discrete_n': 21}
)

gym.envs.register(
     id='StackRGBTrain-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'flatten': False, 'discrete_n': None}
)