import gym

from rgb_stacking.contrib.gym_wrapper import ObservationPreprocess
ACTION_BIN_sIZE = 11

gym.envs.register(
     id='StackRGBTestTriplet-v0',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_random',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletRawDict-v0',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletActorDict-v0',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)


gym.envs.register(
     id='StackRGBTestTripletDict-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTriplet-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1', 'obs_preprocess': True, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTriplet-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2', 'obs_preprocess': True, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTriplet-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3', 'obs_preprocess': True, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTriplet-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4', 'obs_preprocess': True, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTriplet-v5',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5', 'obs_preprocess': True, 'num_discrete_action_bin': 11}
)




gym.envs.register(
     id='StackRGBTestTripletDict-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTestTripletDict-v5',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTrain-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'obs_preprocess': True, 'num_discrete_action_bin': None}
)

gym.envs.register(
     id='StackRGBTrain-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'obs_preprocess': False, 'num_discrete_action_bin': 11}
)

gym.envs.register(
     id='StackRGBTrain-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random', 'obs_preprocess': False, 'num_discrete_action_bin': None}
)