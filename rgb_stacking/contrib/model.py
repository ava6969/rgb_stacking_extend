from typing import Dict

import numpy as np
import torch

from rgb_stacking.contrib.relational_net import ResidualSelfAttention, ResidualSelfAttentionCell
from rgb_stacking.contrib.vision_net import NatureNet
from rgb_stacking.contrib.recurrent_net import RecurrentNet
from rgb_stacking.contrib.arguments import PolicyOption
import gym
from a2c_ppo_acktr.a2c_ppo_acktr.distributions import DiagGaussian, MultiCategorical, Categorical
from rgb_stacking.contrib.common import *


class BasicPolicy(nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Dict,
                 option: PolicyOption,
                 policy: bool,
                 output_layer):
        super(BasicPolicy, self).__init__()

        self.feature_extractor, out_shape = Policy.build_feature_extract(obs_space, option, policy)
        out_size = out_shape[-1]

        if 'post' in option.feature_extract:
            post_t = option.feature_extract['post']
            if post_t == 'sum':
                self.post_process_feature_extract = Sum(1)
            elif post_t == 'mean':
                self.post_process_feature_extract = Mean(1)
            else:
                if 'n_blocks' in option.feature_extract:
                    self.post_process_feature_extract = ResidualSelfAttention(
                        option.feature_extract['n_blocks'],
                        out_shape,
                        option.feature_extract['heads'],
                        option.feature_extract['attn_embed_dim'],
                        True,
                        option.feature_extract['max_pool_out'])
                else:
                    self.post_process_feature_extract = ResidualSelfAttentionCell(
                        out_shape,
                        option.feature_extract['heads'],
                        option.feature_extract['attn_embed_dim'],
                        True,
                        option.feature_extract['max_pool_out'])

        self.rec_net = RecurrentNet(out_size,  option)
        max_action = obs_space['past_action'].high[0]
        self.action_embed = torch.nn.Embedding(int(max_action + 1), 10)
        self.output_layer = output_layer

    def forward(self, inputs: Dict, rnn_hxs, masks):
        output = dict()
        if len(inputs) > 3:
            features = [f_ext(inputs[k]) for k, f_ext in self.feature_extractor.items()]
            output["input"] = self.post_process_feature_extract(torch.stack(features, 1))
        else:
            output["input"] = self.feature_extractor(inputs['observation'])

        prior_joint_action = self.action_embed( inputs['past_action'][:, :-1].long() )
        output['past_action'] = torch.concat( [prior_joint_action.flatten(1),
                                               inputs['past_action'][:, -1].view(-1, 1)], -1)
        output['past_reward'] = inputs['past_reward']
        recurrent_features, rnn_hxs = self.rec_net(output, rnn_hxs, masks, )
        return self.output_layer(recurrent_features), rnn_hxs


class Policy(nn.Module):
    def __init__(self, obs_space: gym.Space, action_space, option: PolicyOption):
        super(Policy, self).__init__()

        self.critic = BasicPolicy(obs_space, option, False,
                                  init_(nn.Linear(option.hidden_size, 1)))

        self.rec_type = option.rec_type
        self._recurrent_hidden_state_size = option.hidden_size if self.is_recurrent else 1

        if action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            dist = MultiCategorical(option.hidden_size, action_space.nvec)
        elif action_space.__class__.__name__ == "Discrete":
            dist = Categorical(option.hidden_size, action_space.n)
        else:
            raise NotImplementedError

        self.actor = BasicPolicy(obs_space, option, True, dist)

    @property
    def is_recurrent(self):
        return self.rec_type is not None

    def zero_state(self, batch_size, rect_type=None):
        f = lambda: torch.zeros(batch_size, self._recurrent_hidden_state_size)
        _type = rect_type if rect_type else self.rec_type

        if _type == "lstm":
            return dict(actor=(f(), f()), critic=(f(), f()))
        else:
            return dict(actor=f(), critic=f())

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self._recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        logit, rnn_hxs['actor'] = self.actor(inputs, rnn_hxs['actor'], masks)
        value, rnn_hxs['critic'] = self.critic(inputs, rnn_hxs['critic'], masks)
        return value, logit, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, dist, rnn_hxs = self.forward(inputs, rnn_hxs, masks)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).view(-1, 1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.forward(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, dist, rnn_hxs = self.forward(inputs, rnn_hxs, masks)

        action_log_probs = dist.log_prob(action).view(-1, 1)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    @staticmethod
    def build_feature_extract(obs_space: gym.spaces.Dict, option: PolicyOption, policy: bool):

        if len(obs_space.spaces) > 3:
            spaces = obs_space.spaces
            embed_dict = torch.nn.ModuleDict()
            for k in (option.policy_keys if policy else option.value_keys):
                if 'reward' not in k and 'action' not in k:
                    space = spaces[k]
                    if len(space.shape) != 3:
                        embed_dict[k] = init_(
                            torch.nn.Linear(int(np.prod(space.shape)), option.feature_extract['embed_out']))
                    else:
                        embed_dict[k] = NatureNet(space.shape, option.feature_extract['vision_out'])
            return embed_dict, [len(embed_dict), option.feature_extract['embed_out']]
        else:
            in_ = np.ravel(obs_space["observation"].shape)[0]
            out_size = option.feature_extract['flatten_out']
            model = torch.nn.Sequential(init_(torch.nn.Linear(in_, out_size)),
                                        torch.nn.ReLU(),
                                        init_(torch.nn.Linear(out_size, out_size)))

            return model, [1, out_size]
