from typing import Dict, List

import gym
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rgb_stacking.utils.mpi_tools import gather


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps=0, num_processes=0, obs_space=None, action_space=None, recurrent_hidden_state_size=0):

        self.recurrent_hidden_states = dict(actor=(None, None), critic=(None, None))
        self.num_steps = num_steps
        self.step = 0

        if num_steps > 0:
            self.rewards = torch.zeros(num_steps, num_processes, 1)
            self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
            self.returns = torch.zeros(num_steps + 1, num_processes, 1)
            self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

            if action_space.__class__.__name__ == 'Discrete':
                action_shape = 1
            else:
                action_shape = action_space.shape[0]
            self.actions = torch.zeros(num_steps, num_processes, action_shape)
            if action_space.__class__.__name__ == 'Discrete':
                self.actions = self.actions.long()

            f = lambda: torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)

            self.is_dict = isinstance(obs_space, gym.spaces.Dict)
            self.recurrent_hidden_states = dict(actor=(f(), f()), critic=(f(), f()))

            self.obs = {k: torch.zeros(num_steps + 1, num_processes, *s.shape)
                        for k, s in obs_space.spaces.items()}

            self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def clone(self):
        return RolloutStorage(0, 0)

    def get_from_recurrent_state(self, step, fn=None):
        actor = self.recurrent_hidden_states['actor']
        critic = self.recurrent_hidden_states['critic']

        return dict(actor=(actor[0][step], actor[1][step]),
                    critic=(critic[0][step], critic[1][step])) if not fn \
            else dict(actor=(fn(actor[0][step]), fn(actor[1][step])), critic=(fn(critic[0][step]),
                                                                              fn(critic[1][step])))

    def append_recurrent_state(self, actor_list, critic_list, fn):
        actor = self.recurrent_hidden_states['actor']
        critic = self.recurrent_hidden_states['critic']
        actor_list.append(tuple([fn(actor[0]), fn(actor[1])]))
        critic_list.append(tuple([fn(critic[0]), fn(critic[1])]))

    def stack_recurrent_state(self, actor_list, critic_list, N):
        fn = lambda x: torch.stack(x, 1).view(N, -1)
        actor_h0, actor_c0 = zip(*actor_list)
        critic_h0, critic_c0 = zip(*critic_list)
        return dict(actor=(fn(actor_h0), fn(actor_c0)), critic=(fn(critic_h0), fn(critic_c0)))

    def get_obs(self, index=None, fn=None):
        return {k: v[index] if fn is None else fn(v) for k, v in self.obs.items()}

    def to(self, device):
        self.obs = {k: v.to(device) for k, v in self.obs.items()}

        for k in self.recurrent_hidden_states:
            self.recurrent_hidden_states[k] = self.recurrent_hidden_states[k][0].to(device), \
                                              self.recurrent_hidden_states[k][1].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):

        if not self.is_dict:
            self.obs[self.step + 1].copy_(obs)
        else:
            for k, v in obs.items():
                self.obs[k][self.step + 1].copy_(v)

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        for k in self.recurrent_hidden_states:
            self.recurrent_hidden_states[k][0][self.step + 1].copy_(recurrent_hidden_states[k][0])
            self.recurrent_hidden_states[k][1][self.step + 1].copy_(recurrent_hidden_states[k][1])

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        if not self.is_dict:
            self.obs[0].copy_(self.obs[-1])
        else:
            for k, v in self.obs.items():
                self.obs[k][0].copy_(v[-1])

        self.masks[0].copy_(self.masks[-1])
        for k in self.recurrent_hidden_states:
            self.recurrent_hidden_states[k][0][self.step + 1].copy_(self.recurrent_hidden_states[k][0][-1])
            self.recurrent_hidden_states[k][1][self.step + 1].copy_(self.recurrent_hidden_states[k][1][-1])


    @staticmethod
    def concat_attr(rollouts: List[object],
                    cat_rollout: object,
                    attr: str,
                    device):

        cat_rollout.__setattr__(attr,
                                torch.cat([ro.__getattribute__(attr).cpu() for ro in rollouts], 1).to(device))

    def mpi_gather(self, comm, device):
        rollouts = gather(comm, self)
        if rollouts:
            cat_rollout = RolloutStorage()

            cat_rollout.obs = {k: torch.cat([r.obs[k].cpu() for r in rollouts], 1).to(device) for k in self.obs.keys()}
            for attr in ['actions', 'action_log_probs', 'value_preds', 'returns', 'rewards', 'masks']:
                self.concat_attr(rollouts, cat_rollout, attr, device)

            cat_rollout.num_steps = self.num_steps

            for k in self.recurrent_hidden_states:
                hs = [ro.recurrent_hidden_states[k][0].cpu() for ro in rollouts]
                cs = [ro.recurrent_hidden_states[k][1].cpu() for ro in rollouts]
                cat_rollout.recurrent_hidden_states[k] = torch.cat(hs, 1).to(device), torch.cat(cs, 1).to(device)

            return cat_rollout
        return None

    def compute_returns(self, next_value, gamma, gae_lambda):

        self.value_preds[-1] = next_value
        gae = 0

        for step in reversed(range(self.rewards.size(0))):
            discounted = gamma * self.value_preds[step + 1] * self.masks[step + 1]

            delta = self.rewards[step] + discounted - self.value_preds[step]

            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

            self.returns[step] = gae + self.value_preds[step]

    @staticmethod
    def apply_to_dict_obs(obs, fn):
        _out = dict()
        for k, v in obs.items():
            _out[k] = fn(v)
        return _out

    def apply_in_place_to_dict_obs(self, fn):
        for k, v in self.obs.items():
            fn(k, v)

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = dict({k: [] for k in self.obs})
            recurrent_hidden_states_batch = [[], []]
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                self.apply_in_place_to_dict_obs(lambda k, x: obs_batch[k].append(x[:-1, ind]))
                self.append_recurrent_state(recurrent_hidden_states_batch[0],
                                            recurrent_hidden_states_batch[1],
                                            lambda x: x[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = self.apply_to_dict_obs(obs_batch, lambda x: torch.stack(x, 1))
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            recurrent_hidden_states_batch = self.stack_recurrent_state(recurrent_hidden_states_batch[0],
                                                                       recurrent_hidden_states_batch[1], N)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = self.apply_to_dict_obs(obs_batch, lambda v: v.view(T * N, *v.size()[2:]))
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
