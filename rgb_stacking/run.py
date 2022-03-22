import logging
import os

from rgb_stacking.utils.mpi_pytorch import sync_params, learner_group, MPI
from rgb_stacking.utils.mpi_tools import proc_id, num_procs, gather, msg

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from collections import deque
from typing import Sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import time
from rgb_stacking.a2c_ppo_acktr import utils
from rgb_stacking.contrib import algo
from rgb_stacking.contrib.arguments import get_args
from rgb_stacking.contrib.envs import make_vec_envs
from rgb_stacking.contrib.model import Policy
from rgb_stacking.contrib.storage import RolloutStorage
from rgb_stacking.a2c_ppo_acktr.evaluation import evaluate
from absl import app
from absl import flags
from absl.flags import FLAGS

flags.DEFINE_string('config_path',
                    "/home/ava6969/rgb_stacking_extend/rgb_stacking/contrib/configs/CONFIG_A.yaml",
                    'path to config')


def make_eval_envs(main_env):
    return []


logging.disable(logging.CRITICAL)
def main(argv: Sequence[str]) -> None:

    env = os.environ.copy()

    env.update(
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        IN_MPI="1"
    )

    args = get_args(FLAGS.config_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device != 'cpu' and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if proc_id() == 0:
        writer = SummaryWriter()

    torch.set_num_threads(1)
    device = torch.device(args.device)

    envs = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_envs_per_cpu,
                         args.gamma,
                         device,
                         args.use_multi_thread)

    actor_critic = Policy(envs.observation_space, envs.action_space, args.model)
    actor_critic.to(device)

    if proc_id() == 0:
        print(actor_critic)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.entropy_coef,
            args.value_loss_coef,
            vlr=args.vlr,
            plr=args.plr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            vlr=args.vlr,
            plr=args.plr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_envs_per_cpu,
                              envs.observation_space, envs.action_space,
                              args.model.hidden_size,
                              args.model.rec_type)

    avg_grad_comm, rollout_per_learner_comm = learner_group(args.num_learners)
    msg('avg_grad_group(rank={}, size={}), '
        'rollout_per_learner_group(rank={}, size={})'.format(avg_grad_comm.rank if avg_grad_comm else 'null',
                                                             avg_grad_comm.size if avg_grad_comm else 'null',
                                                             rollout_per_learner_comm.rank if rollout_per_learner_comm else 'null',
                                                             rollout_per_learner_comm.size if rollout_per_learner_comm else 'null'))

    obs = envs.reset()

    if isinstance(obs, dict):
        for k, o in obs.items():
            rollouts.obs[k][0].copy_(o)
    else:
        rollouts.obs[0].copy_(obs)

    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    start = time.time()
    num_updates = int(
        args.num_env_steps * num_procs() * args.num_envs_per_cpu) // args.num_steps

    for j in range(num_updates):

        sync_params(actor_critic)
        actor_critic.eval()
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if agent.shared_model:
                p_lr_ = utils.update_linear_schedule(
                    agent.actor_critic_optimizer, j, num_updates,
                    agent.actor_critic_optimizer.lr if args.algo == "acktr" else args.plr)
                if proc_id() == 0:
                    writer.add_scalar('LearningRate/ActorCritic', p_lr_, j)
            else:
                p_lr_ = utils.update_linear_schedule(
                    agent.actor_optimizer, j, num_updates,
                    agent.actor_optimizer.lr if args.algo == "acktr" else args.plr)
                v_lr_ = utils.update_linear_schedule(
                    agent.critic_optimizer, j, num_updates,
                    agent.critic_optimizer.lr if args.algo == "acktr" else args.vlr)
                if proc_id() == 0:
                    writer.add_scalar('LearningRate/Critic', v_lr_, j)
                    writer.add_scalar('LearningRate/Actor', p_lr_, j)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.get_obs(step),
                    rollouts.get_from_recurrent_state(step),
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.get_obs(-1), rollouts.get_from_recurrent_state(-1),
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

        cat_rollout = rollouts.mpi_gather(rollout_per_learner_comm, args.device)

        actor_critic.train()

        if cat_rollout is not None:
            value_loss, action_loss, dist_entropy = agent.update(avg_grad_comm, cat_rollout, args.num_learners)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        _group_episode_rewards = gather(MPI.COMM_WORLD, episode_rewards)
        group_episode_rewards = []
        if proc_id() == 0:
            for x in _group_episode_rewards:
                group_episode_rewards.extend(x)

            if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))

            if j % args.log_interval == 0 and len(group_episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_steps * args.num_envs_per_cpu * num_procs()
                end = time.time()
                fps = int(total_num_steps / (end - start))

                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} "
                    "training episodes: mean/median reward {:.1f}/{:.1f}, "
                    "min/max reward {:.1f}/{:.1f}\n ".format(j, total_num_steps,
                                                             fps,
                                                             len(group_episode_rewards), np.mean(group_episode_rewards),
                                                             np.median(group_episode_rewards),
                                                             np.min(group_episode_rewards),
                                                             np.max(group_episode_rewards), dist_entropy, value_loss,
                                                             action_loss))

                writer.add_scalar('Reward/Mean', float(np.mean(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Reward/Min', float(np.min(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Reward/Median', float(np.median(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Reward/Max', float(np.max(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Loss/Policy', action_loss, total_num_steps)
                writer.add_scalar('Loss/Value', value_loss, total_num_steps)
                writer.add_scalar('Loss/Entropy', dist_entropy, total_num_steps)
                writer.add_scalar('Timing/FPS', fps, total_num_steps)

            if args.eval_interval is not None and len(group_episode_rewards) > 1 and j % args.eval_interval == 0:
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(actor_critic, obs_rms, make_eval_envs(args.env_name),
                         args.seed, 5, eval_log_dir, device)

            writer.add_scalar('Timing/Updates', j, j)

    envs.close()
    writer.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
