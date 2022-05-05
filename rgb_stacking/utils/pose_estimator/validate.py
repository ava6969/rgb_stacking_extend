import gym
import numpy as np
from absl import flags, app
from absl.flags import FLAGS
import torch
import tensorflow as tf
from dm_control import viewer


from rgb_stacking.utils import environment, policy_loading
from rgb_stacking.utils.caliberate import relative_pose, max_workspace_limits
from rgb_stacking.utils.policy_loading import policy_from_path
from rgb_stacking.utils.pose_estimator.model import LargeVisionModule, VisionModule


def main(_argv):


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    pose_estimator = torch.load('/home/dewe/rgb_stacking_extend/small_model_256.pt', map_location='cuda')

    pose_estimator.eval()

    policy = policy_from_path(
        '/home/dewe/rgb_stacking_extend/rgb_stacking/utils/assets/saved_models/mpo_state_rgb_test_triplet2')

    policy = policy_loading.StatefulPolicyCallable(policy)

    with environment.rgb_stacking(object_triplet='rgb_test_triplet2',
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True) as env:

        def run(timestep):

            obs = timestep.observation
            r, g, b = obs['rgb30_red/abs_pose'][-1],  obs['rgb30_green/abs_pose'][-1],  obs['rgb30_blue/abs_pose'][-1]
            r_rel, g_rel, b_rel = relative_pose(r), relative_pose(g), relative_pose(b)
            r[:3], g[:3], b[:3] = r_rel[:3] / 0.25, g_rel[:3] / 0.25, b_rel[:3] / 0.25

            cams = [ obs['basket_back_left/pixels'], obs['basket_front_left/pixels'], obs['basket_front_right/pixels']]
            cams = np.stack(cams)
            cams = torch.from_numpy(cams).permute(0, 3, 1, 2) / 255

            with torch.no_grad():
                prediction = pose_estimator( cams.unsqueeze(0).cuda() )

            # print('prediction: \nr: {}\ng:{}\nb:{}'.format(
            #     *[ l.cpu().numpy()  for l in torch.unbind( prediction ) ] ) )
            #
            # print('label: \nr:{}\ng:{}\nb:{}'.format(r, g, b))

            print('r_pos_loss: {}\t'.format( np.square(prediction[0].cpu().numpy() - r[:3]).mean() ),
            'r_orient_loss: {}'.format(np.square(prediction[1].cpu().numpy() - r[3:]).mean()),
            'g_pos_loss: {}'.format(np.square(prediction[2].cpu().numpy() - g[:3]).mean()),
            'g_orient_loss: {}'.format(np.square(prediction[3].cpu().numpy() - g[3:]).mean()),
            'b_pos_loss: {}'.format(np.square(prediction[4].cpu().numpy() - b[:3]).mean()),
            'b_orient_loss: {}'.format(np.square(prediction[5].cpu().numpy() - b[3:]).mean()))

            return policy(timestep)

        viewer.launch(env, policy=run)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass