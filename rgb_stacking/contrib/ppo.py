import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rgb_stacking.utils.mpi_pytorch import mpi_avg_grads


class PPO:

    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, entropy_coef,
                 value_loss_coef=None, vlr=None, plr=None, eps=None, max_grad_norm=None):

        self.actor = actor_critic.actor
        self.critic = actor_critic.critic
        self.actor_critic = actor_critic

        self.vf_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.actor_optimizer = optim.Adam(self.actor_critic.parameters(), plr, eps=eps)
        self.critic_optimizer = optim.Adam(self.actor_critic.parameters(), vlr, eps=eps)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

    def update_model(self, COMM, opt, loss, model, num_learners):
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        mpi_avg_grads(model, COMM, num_learners)
        opt.step()

    def update_actor_critic(self, COMM, action_loss, value_loss, dist_entropy, num_learners):
        self.update_model(COMM,
                          self.actor_optimizer, action_loss - self.entropy_coef * dist_entropy, self.actor,
                          num_learners)
        self.update_model(COMM,
                          self.critic_optimizer, value_loss, self.critic, num_learners)

    def update(self, COMM, rollouts, num_learners):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = - torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.update_actor_critic(COMM, action_loss, value_loss, dist_entropy, num_learners)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
