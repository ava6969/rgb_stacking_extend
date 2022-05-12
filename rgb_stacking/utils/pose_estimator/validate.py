import numpy as np
from absl import app
import torch
import tensorflow as tf
from dm_control import viewer


from rgb_stacking.utils import environment, policy_loading
from rgb_stacking.utils.caliberate import relative_pose
from rgb_stacking.utils.policy_loading import policy_from_path


def main(_argv):


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    pose_estimator = torch.load('/home/dewe/rgb_stacking_extend/largeno_drskl-a-48.rc.rit.edu_model_64.pt', map_location='cuda:1')

    pose_estimator.eval()

    policy = policy_from_path(
        '/home/dewe/rgb_stacking_extend/rgb_stacking/utils/assets/saved_models/mpo_state_rgb_test_triplet2')

    policy = policy_loading.StatefulPolicyCallable(policy)

    with environment.rgb_stacking(object_triplet='rgb_test_triplet2',
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True, frameStack=3) as env:

        def run(timestep):

            obs = timestep.observation
            r, g, b = obs['rgb30_red/abs_pose'][-1],  obs['rgb30_green/abs_pose'][-1],  obs['rgb30_blue/abs_pose'][-1]

            cams = dict(bl=obs['basket_back_left/pixels'],
                        fl=obs['basket_front_left/pixels'],
                        fr=obs['basket_front_right/pixels'])
            cams = {k: torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).to('cuda:1') / 255 for k, l in cams.items() }

            with torch.no_grad():
                prediction = pose_estimator( cams ).view(-1)

            # print('prediction: \nr: {}\ng:{}\nb:{}'.format(
            #     *[ l.cpu().numpy()  for l in torch.unbind( prediction ) ] ) )
            #
            # print('label: \nr:{}\ng:{}\nb:{}'.format(r, g, b))

            print('r_pos_loss: {}\t'.format( np.square(prediction[:3].cpu().numpy() - r[:3]).mean() ),
            'r_orient_loss: {}'.format(np.square(prediction[3:7].cpu().numpy() - r[3:]).mean()),
            f'r_pos_pred: {prediction[:3].cpu().numpy()}, r_pos_actual: {r[:3]}',
            f'r_pos_orient: {prediction[3:7].cpu().numpy()}, r_pos_actual: {r[3:7]}',
            # 'g_pos_loss: {}'.format(np.square(prediction[14:17].cpu().numpy() - g[:3]).mean()),
            # 'g_orient_loss: {}'.format(np.square(prediction[17:21].cpu().numpy() - g[3:]).mean()),
            # 'b_pos_loss: {}'.format(np.square(prediction[7:10].cpu().numpy() - b[:3]).mean()),
            # 'b_orient_loss: {}'.format(np.square(prediction[10:14].cpu().numpy() - b[3:]).mean())
                  )

            return policy(timestep)

        viewer.launch(env, policy=run)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass