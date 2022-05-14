import numpy as np
from absl import app
import torch
import tensorflow as tf
from dm_control import viewer

import rgb_stacking.utils.pose_estimator.model
from rgb_stacking.utils import environment, policy_loading
from rgb_stacking.utils.caliberate import relative_pose
from rgb_stacking.utils.policy_loading import policy_from_path


def main(_argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # pose_estimator = torch.load('coreno_drskl-a-48.rc.rit.edu_model_64.pt', map_location='cuda:1')
    # torch.save(pose_estimator.state_dict(), 'pose_estimator.pt')

    pe_state_dict = torch.load('pose_estimator.pt')
    pose_estimator = rgb_stacking.utils.pose_estimator.model.VisionModule().cuda()
    pose_estimator.load_state_dict(pe_state_dict)
    pose_estimator.eval()

    policy = policy_from_path(
        'rgb_stacking/utils/assets/saved_models/mpo_state_rgb_test_triplet1')

    policy = policy_loading.StatefulPolicyCallable(policy)

    global rPos_error, gPos_error, bPos_error, rOrient_error, gOrient_error, bOrient_error, total
    rPos_error, gPos_error, bPos_error = 0, 0, 0
    rOrient_error, gOrient_error, bOrient_error = 0, 0, 0
    total = 0

    with environment.rgb_stacking(object_triplet='rgb_test_triplet1',
                                  observation_set=environment.ObservationSet.ALL,
                                  use_sparse_reward=True, frameStack=3) as env:
        def run(timestep):
            global rPos_error, gPos_error, bPos_error, rOrient_error, gOrient_error, bOrient_error, total
            obs = timestep.observation
            r, g, b = obs['rgb30_red/abs_pose'][-1], obs['rgb30_green/abs_pose'][-1], obs['rgb30_blue/abs_pose'][-1]

            cams = dict(bl=obs['basket_back_left/pixels'],
                        fl=obs['basket_front_left/pixels'],
                        fr=obs['basket_front_right/pixels'])
            cams = {k: torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).cuda() / 255 for k, l in cams.items()}

            with torch.no_grad():
                prediction = pose_estimator(cams).view(-1)

            rPos_error += (prediction[:3].cpu().numpy() - r[:3]).sum()
            gPos_error += (prediction[14:17].cpu().numpy() - g[:3]).sum()
            bPos_error += (prediction[7:10].cpu().numpy() - b[:3]).sum()
            rOrient_error += (prediction[3:7].cpu().numpy() - r[3:]).sum()
            gOrient_error += (prediction[17:21].cpu().numpy() - g[3:]).sum()
            bOrient_error += (prediction[10:14].cpu().numpy() - b[3:]).sum()
            # print('r_pos_loss: {}\t'.format(np.square(prediction[:3].cpu().numpy() - r[:3]).mean()),
            #       'r_orient_loss: {}'.format(np.square(prediction[3:7].cpu().numpy() - r[3:]).mean()),
            #       f'r_pos_pred: {prediction[:3].cpu().numpy()}, r_pos_actual: {r[:3]}',
            #       f'r_pos_orient: {prediction[3:7].cpu().numpy()}, r_pos_actual: {r[3:7]}',
            #       # 'g_pos_loss: {}'.format(np.square(prediction[14:17].cpu().numpy() - g[:3]).mean()),
            #       # 'g_orient_loss: {}'.format(np.square(prediction[17:21].cpu().numpy() - g[3:]).mean()),
            #       # 'b_pos_loss: {}'.format(np.square(prediction[7:10].cpu().numpy() - b[:3]).mean()),
            #       # 'b_orient_loss: {}'.format(np.square(prediction[10:14].cpu().numpy() - b[3:]).mean())
            #       )
            total += 1
            return policy(timestep)

        viewer.launch(env, policy=run)
        print('rPos_mean_abs_error', abs(rPos_error/total/3) )
        print('rOrient_mean_abs_error', abs(rOrient_error/total/4) )
        print('gPos_mean_abs_error', abs(gPos_error/total/3) )
        print('gOrient_mean_abs_error', abs(gOrient_error / total / 4))
        print('bPos_mean_abs_error', abs(bPos_error / total / 3))
        print('bOrient_mean_abs_error', abs(bOrient_error / total / 4))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
