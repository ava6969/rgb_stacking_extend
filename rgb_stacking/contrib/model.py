from typing import Dict

import torch

from rgb_stacking.contrib.relational_net import ResidualSelfAttention, ResidualSelfAttentionCell
from rgb_stacking.contrib.vision_net import RobotImageResnetModule
from rgb_stacking.contrib.recurrent_net import RecurrentNet
from rgb_stacking.contrib.arguments import PolicyOption
import gym
from rgb_stacking.utils.distributions import DiagGaussian, MultiCategorical, Categorical
from rgb_stacking.contrib.common import *


class BasicPolicy(nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Dict,
                 option: PolicyOption,
                 model_type: str,
                 output_layer):
        super(BasicPolicy, self).__init__()

        self.feature_extractor, out_shape = Policy.build_feature_extract(obs_space, option, model_type)
        out_size = out_shape[-1]
        self.dict_obs = False
        if 'post' in option.feature_extract:
            post_t = option.feature_extract['post']

            if post_t == 'sum':
                self.post_process_feature_extract = Sum(1)
            elif post_t == 'mean':
                self.post_process_feature_extract = Mean(1)
            elif post_t == 'attn':
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
                self.dict_obs = True
            else:
                raise TypeError('post_t only supports [attn|sum|mean] not {}'.format(post_t))
        else:
            self.model_type = model_type

        self.rec_net = RecurrentNet(out_size + obs_space.spaces['past_action'].shape[0],
                                    obs_space.spaces['past_action'].shape[0],
                                    option)
        self.output_layer = output_layer

    def forward(self, inputs, action, rnn_hxs, masks):
        output = [action]
        if self.dict_obs:
            features = [f_ext(inputs[k]) for k, f_ext in self.feature_extractor.items()]
            output.append(self.post_process_feature_extract(torch.stack(features, 1)))
        else:
            output.append(self.feature_extractor(inputs[self.model_type]))

        recurrent_features, rnn_hxs = self.rec_net(torch.cat(output, -1), rnn_hxs, masks, )
        return self.output_layer(recurrent_features), rnn_hxs


class Policy(nn.Module):
    def __init__(self, obs_space: gym.spaces.Dict, action_space, option: PolicyOption):
        super(Policy, self).__init__()

        self._recurrent_hidden_state_size = option.hidden_size
        dist = MultiCategorical(option.hidden_size, action_space.nvec)
        self.critic = BasicPolicy(obs_space, option, 'critic', init_(nn.Linear(option.hidden_size, 1)))
        self.actor = BasicPolicy(obs_space, option, 'actor', dist)

    @property
    def is_recurrent(self):
        return self.rec_type is not None

    def zero_state(self, batch_size, rect_type=None):
        f = lambda: torch.zeros(batch_size, self._recurrent_hidden_state_size)
        _type = rect_type if rect_type else self.rec_type

        return dict(actor=(f(), f()), critic=(f(), f()))

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self._recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        action = inputs['past_action']
        logit, rnn_hxs['actor'] = self.actor(inputs, action, rnn_hxs['actor'], masks)
        value, rnn_hxs['critic'] = self.critic(inputs, action, rnn_hxs['critic'], masks)
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
    def build_feature_extract(obs_space: gym.spaces.Dict,
                              option: PolicyOption,
                              model_type: str):

        if 'post' in option.feature_extract:
            spaces = obs_space.spaces
            embed_dict = torch.nn.ModuleDict()
            for k, space in spaces.items():
                if k.startswith(model_type):
                    embed_dict[k] = init_(torch.nn.Linear(int(np.prod(space.shape)),
                                                          option.feature_extract['embed_out']))
            return embed_dict, [len(embed_dict), option.feature_extract['embed_out']]
        else:
            in_ = np.ravel(obs_space[model_type].shape)[0]
            out_size = option.feature_extract['flatten_out']
            model = torch.nn.Sequential(init_(torch.nn.Linear(in_, out_size)),
                                        torch.nn.ReLU(),
                                        init_(torch.nn.Linear(out_size, out_size)))
            return model, [1, out_size]
