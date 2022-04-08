from rgb_stacking.utils.mpi_pytorch import sync_params, learner_group, MPI
from rgb_stacking.utils.mpi_tools import proc_id, num_procs, gather, msg
import logging
import os
import socket
from collections import deque
from typing import Sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import time
from rgb_stacking.utils import utils
from rgb_stacking.contrib import ppo
from rgb_stacking.contrib.arguments import get_args
from rgb_stacking.contrib.envs import make_vec_envs, VecPyTorch
from rgb_stacking.contrib.model import Policy
from rgb_stacking.contrib.storage import RolloutStorage
from rgb_stacking.utils.evaluation import evaluate
from absl import app
from absl import flags
from absl.flags import FLAGS


flags.DEFINE_string('config_path', "", 'path to config')
logging.disable(logging.CRITICAL)


def init_env():
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        IN_MPI="1"
    )
    return env


def _seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)


def _mpi_init():
    MPI.Init()
    msg('Successfully loaded')


def to_device(args, envs):
    if args.device == 'infer':
        n_devices = torch.cuda.device_count()
        device_id = proc_id() % n_devices
        args.device = 'cuda:{}'.format(device_id)
        _cuda = torch.cuda.device(args.device)
        msg('{}, {}\t{}'.format(socket.gethostname(),
                                str(torch.cuda.get_device_properties(_cuda)), str(args.device)))

    device = torch.device(args.device)
    envs = VecPyTorch(envs, device)

    return device, envs


def mpi_groups(args):
    avg_grad_comm, rollout_per_learner_comm = learner_group(args.num_learners)
    msg('avg_grad_group(rank={}, size={}), '
        'rollout_per_learner_group(rank={}, size={})'
        .format(avg_grad_comm.rank if avg_grad_comm else 'null',
                avg_grad_comm.size if avg_grad_comm else 'null',
                rollout_per_learner_comm.rank if rollout_per_learner_comm else 'null',
                rollout_per_learner_comm.size if rollout_per_learner_comm else 'null'))
    return avg_grad_comm, rollout_per_learner_comm


def run(args, envs, policy, agent, rollouts, writer, device, rollout_per_learner_comm, avg_grad_comm):
    save_path = os.path.join(args.save_dir)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    episode_rewards = deque(maxlen=100)
    best_reward = -np.inf
    start = time.time()
    num_updates = int(args.num_env_steps * num_procs() * args.num_envs_per_cpu) // args.num_steps
    value_loss, action_loss, dist_entropy = None, None, None

    for j in range(num_updates):
        sync_params(policy)
        policy.eval()

        p_lr_ = utils.update_linear_schedule(agent.actor_optimizer, j, num_updates, args.plr)
        v_lr_ = utils.update_linear_schedule(agent.critic_optimizer, j, num_updates, args.vlr)

        if writer:
            writer.add_scalar('LearningRate/Critic', v_lr_, j)
            writer.add_scalar('LearningRate/Actor', p_lr_, j)

        for step in range(args.num_steps):

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.get_obs(step),
                    rollouts.get_from_recurrent_state(step),
                    rollouts.masks[step])

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
            next_value = policy.get_value(
                rollouts.get_obs(-1), rollouts.get_from_recurrent_state(-1),
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        cat_rollout = rollouts.mpi_gather(rollout_per_learner_comm, args.device)

        policy.train()

        if cat_rollout is not None:
            value_loss, action_loss, dist_entropy, dist_kl = agent.update(avg_grad_comm, cat_rollout, args.num_learners)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        _group_episode_rewards = gather(MPI.COMM_WORLD, episode_rewards)
        group_episode_rewards = []
        if proc_id() == 0:
            for x in _group_episode_rewards:
                group_episode_rewards.extend(x)

            mean_rew = -np.inf
            if len(group_episode_rewards) > 1:
                mean_rew = float(np.mean(group_episode_rewards))

            if mean_rew > best_reward:
                print('Saved Best Model @ mean_100_reward:', mean_rew)
                torch.save([policy, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)],
                           os.path.join(save_path, args.env_name + "best.pt"))
                best_reward = mean_rew

            if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
                torch.save([
                    policy,
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
                                                             len(group_episode_rewards), mean_rew,
                                                             np.median(group_episode_rewards),
                                                             np.min(group_episode_rewards),
                                                             np.max(group_episode_rewards), dist_entropy, value_loss,
                                                             action_loss))

                writer.add_scalar('Reward/Mean', mean_rew, total_num_steps)
                writer.add_scalar('Reward/Min', float(np.min(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Reward/Median', float(np.median(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Reward/Max', float(np.max(group_episode_rewards)), total_num_steps)
                writer.add_scalar('Loss/Policy', action_loss, total_num_steps)
                writer.add_scalar('Loss/Value', value_loss, total_num_steps)
                writer.add_scalar('Loss/Entropy', dist_entropy, total_num_steps)
                writer.add_scalar('Loss/KL', dist_kl, total_num_steps)
                writer.add_scalar('Timing/FPS', fps, total_num_steps)

            if args.eval_interval is not None and len(group_episode_rewards) > 1 and j % args.eval_interval == 0:
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(policy, obs_rms, args.env_name, args.seed, 5, None, device)

            writer.add_scalar('Timing/Updates', j, j)


def main(argv: Sequence[str]) -> None:
    init_env()

    args = get_args(FLAGS.config_path)

    _seed(args)

    envs = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_envs_per_cpu,
                         args.gamma,
                         args.use_multi_thread)

    _mpi_init()

    device, envs = to_device(args, envs)

    if num_procs() > 1:
        args.num_env_steps = args.num_env_steps // num_procs()
        args.seed += 10000 * proc_id()

    writer = None
    if proc_id() == 0:
        writer = SummaryWriter()

    actor_critic = Policy(envs.observation_space, envs.action_space, args.model)
    actor_critic.to(device)
    envs.seed(args.seed)

    if proc_id() == 0:
        print(actor_critic)

    agent = ppo.PPO(
        actor_critic=actor_critic,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        entropy_coef=args.entropy_coef,
        vlr=args.vlr,
        plr=args.plr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_envs_per_cpu,
                              envs.observation_space, envs.action_space,
                              args.model.hidden_size)

    avg_grad_comm, rollout_per_learner_comm = mpi_groups(args)

    obs = envs.reset()
    for k, o in obs.items():
        rollouts.obs[k][0].copy_(o)

    rollouts.to(device)

    run(args=args, envs=envs, policy=actor_critic,
        agent=agent, rollouts=rollouts, writer=writer, device=device,
        rollout_per_learner_comm=rollout_per_learner_comm,
        avg_grad_comm=avg_grad_comm)

    envs.close()

    if writer:
        writer.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
