import torch
import torch.nn as nn
import torch.optim as optim

from a2c_ppo_acktr.a2c_ppo_acktr.algo.kfac import KFACOptimizer
# from rgb_stacking.contrib.mpi_pytorch import mpi_avg_grads


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 entropy_coef,
                 vf_coef=None,
                 vlr=None,
                 plr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor = actor_critic.actor
        self.critic = actor_critic.critic
        self.actor_critic = actor_critic

        self.acktr = acktr
        self.vf_coef = vf_coef
        self.shared_model = self.actor_critic.image_net
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        optim_fn = lambda _m, _lr: KFACOptimizer(_m) if acktr \
            else optim.RMSprop(_m.parameters(), _lr, eps=eps, alpha=alpha) \
            if alpha else optim.Adam(_m.parameters(), _lr, eps=eps)

        if self.shared_model:
            self.actor_critic_optimizer = optim_fn(self.actor_critic, plr)
        else:
            self.actor_optimizer = optim_fn(self.actor, plr)
            self.critic_optimizer = optim_fn(self.critic, vlr)

    def kfac_update(self, action_log_probs, values):
        if self.shared_model:
            self.actor_critic_optimizer.zero_grad()
        else:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

        pg_fisher_loss = -action_log_probs.mean()

        value_noise = torch.randn(values.size())
        if values.is_cuda:
            value_noise = value_noise.cuda()

        sample_values = values + value_noise
        vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

        if self.shared_model:
            self.actor_critic_optimizer.acc_stats = True
            (pg_fisher_loss + vf_fisher_loss).backward(retain_graph=True)
            self.actor_critic_optimizer.acc_stats = False
        else:
            self.actor_optimizer.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            self.actor_optimizer.acc_stats = False

            self.critic_optimizer.acc_stats = True
            vf_fisher_loss.backward(retain_graph=True)
            self.critic_optimizer.acc_stats = False

    def update_model(self, opt, loss, model):
        opt.zero_grad()
        loss.backward()
        if not self.acktr:
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        # mpi_avg_grads(model)
        opt.step()

    def update_actor_critic(self, action_loss, value_loss, dist_entropy):
        if self.shared_model:
            self.update_model(self.actor_critic_optimizer,
                              action_loss - self.entropy_coef * dist_entropy + self.vf_coef * value_loss,
                              self.actor_critic)
        else:
            self.update_model(self.actor_optimizer, action_loss - self.entropy_coef * dist_entropy, self.actor)
            self.update_model(self.critic_optimizer, value_loss, self.critic)

    def update(self, rollouts):

        action_shape = rollouts.actions.size()[-1]

        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.get_obs(None, lambda x: x[:-1].view(-1, *(x.size()[2:]))),
            rollouts.get_from_recurrent_state(0, lambda x: x.view(-1, self.actor_critic.recurrent_hidden_state_size)),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and \
                ((self.shared_model and self.actor_critic_optimizer.steps % self.actor_critic_optimizer.Ts) or
                 (not self.shared_model and self.actor_optimizer.steps % self.actor_optimizer.Ts == 0)):
            self.kfac_update(action_log_probs, values)
        else:
            self.update_actor_critic(action_loss, value_loss, dist_entropy)

        return value_loss.item(), action_loss.item(), dist_entropy.item()
