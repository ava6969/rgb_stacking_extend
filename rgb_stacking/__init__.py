import gym

from rgb_stacking.contrib.gym_wrapper import ObservationPreprocess
ACTION_BIN_sIZE = 11

''' BEGIN StackRGBTestTriplet-v0 '''
gym.envs.register(
     id='StackRGBTestTriplet-v0',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet1',
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
''' END StackRGBTestTriplet-v0 '''

gym.envs.register(
     id='StackRGBTestTriplet-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletRawDict-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletActorDict-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet2',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTriplet-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletRawDict-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletActorDict-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet3',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTriplet-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletRawDict-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletActorDict-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet4',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTriplet-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletRawDict-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTestTripletActorDict-v4',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_test_triplet5',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)


gym.envs.register(
     id='StackRGBTrain-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTrain-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTrain-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE}
)

gym.envs.register(
     id='StackRGBTrainImage-v1',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.FLATTEN,
             'num_discrete_action_bin': ACTION_BIN_sIZE,
             'add_image': True}
)

gym.envs.register(
     id='StackRGBTrainImage-v2',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.RAW_DICT,
             'num_discrete_action_bin': ACTION_BIN_sIZE,
             'add_image': True}
)

gym.envs.register(
     id='StackRGBTrainImage-v3',
     entry_point='contrib.gym_wrapper:GymWrapper',
     max_episode_steps=400,
     kwargs={'object_triplet': 'rgb_train_random',
             'obs_preprocess': ObservationPreprocess.ACTOR_BASED,
             'num_discrete_action_bin': ACTION_BIN_sIZE,
             'add_image': True}
)

