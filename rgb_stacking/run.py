
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
from collections import deque
from typing import Sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from a2c_ppo_acktr.a2c_ppo_acktr import utils
import a2c_ppo_acktr.a2c_ppo_acktr.algo as shared_algo
from rgb_stacking.contrib import algo
from rgb_stacking.contrib.arguments import get_args
from a2c_ppo_acktr.a2c_ppo_acktr.envs import make_vec_envs
from rgb_stacking.contrib.model import Policy
from rgb_stacking.contrib.storage import RolloutStorage
from a2c_ppo_acktr.evaluation import evaluate

from absl import app
from absl import flags
from absl.flags import FLAGS


flags.DEFINE_string('config_path',
                    "/home/ava6969/rgb_stacking_extend/rgb_stacking/contrib/configs/CONFIG_A.yaml",
                    'path to config')


def main(argv: Sequence[str]) -> None:

    args = get_args(FLAGS.config_path)
    writer = SummaryWriter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(envs.observation_space, envs.action_space, args.model)

    actor_critic.to(device)
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
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space,
                              args.model.hidden_size,
                              args.model.rec_type)

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
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        actor_critic.eval()
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if agent.shared_model:
                utils.update_linear_schedule(
                    agent.actor_critic_optimizer, j, num_updates,
                    agent.actor_critic_optimizer.plr if args.algo == "acktr" else args.plr)
                writer.add_scalar('LearningRate/ActorCritic', agent.actor_critic_optimizer.plr, j)
            else:
                utils.update_linear_schedule(
                    agent.actor_optimizer, j, num_updates,
                    agent.actor_optimizer.plr if args.algo == "acktr" else args.plr)
                utils.update_linear_schedule(
                    agent.critic_optimizer, j, num_updates,
                    agent.critic_optimizer.vlr if args.algo == "acktr" else args.vlr)
                writer.add_scalar('LearningRate/Critic', agent.critic_optimizer.vlr, j)
                writer.add_scalar('LearningRate/Actor', agent.actor_optimizer.plr, j)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.get_obs(step), rollouts.get_from_recurrent_state(step),
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

        actor_critic.train()

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            fps = int(total_num_steps / (end - start))
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n "
                    .format(j, total_num_steps,
                            fps,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

            writer.add_scalar('Reward/Mean', float(np.mean(episode_rewards)), total_num_steps)
            writer.add_scalar('Reward/Min', float(np.min(episode_rewards)), total_num_steps)
            writer.add_scalar('Reward/Median', float(np.median(episode_rewards)), total_num_steps)
            writer.add_scalar('Reward/Max', float(np.max(episode_rewards)), total_num_steps)
            writer.add_scalar('Timing/FPS', fps, total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        writer.add_scalar('Timing/Updates', j, j)

    envs.close()
    writer.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
