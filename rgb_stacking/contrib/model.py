import numpy as np
import torch

from rgb_stacking.contrib.arguments import PolicyOption
import gym
from a2c_ppo_acktr.a2c_ppo_acktr.distributions import DiagGaussian, MultiCategorical, Categorical
from rgb_stacking.contrib.common import *


class BasicPolicy(nn.Module):
    def __init__(self,
                 obs_space: gym.Space,
                 option: PolicyOption,
                 policy: bool,
                 output_layer):
        super(BasicPolicy, self).__init__()

        self.feature_extractor, out_size = build_feature_extract(obs_space, option, policy)

        if isinstance(obs_space, gym.spaces.Dict):
            post_t = option.feature_extract['post']
            self.post_process_feature_extract = Sum(1) if post_t == 'sum' else Mean(1) if post_t == 'mean' else \
                ResidualSelfAttentionBlock(out_size,
                                           option.feature_extract['heads'],
                                           option.feature_extract['embed_dim'],
                                           option.feature_extract['layer_norm'],
                                           option.feature_extract['post_layer_norm'])

        self.rec_net = RecurrentNet(out_size, option)
        self.output_layer = output_layer

    def forward(self, inputs, rnn_hxs, masks):
        if isinstance(inputs, dict):
            features = []
            for k, feature in inputs.items():
                features.append(self.feature_extractor[k](feature))

            inputs = self.post_process_feature_extract(torch.stack(features, 1))
        else:
            inputs = self.feature_extractor(inputs)

        recurrent_features, rnn_hxs = self.rec_net(inputs, rnn_hxs, masks)
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


class RecurrentNet(nn.Module):
    def __init__(self, input_size, option: PolicyOption):
        super(RecurrentNet, self).__init__()
        base_fn = torch.nn.LSTM if option.rec_type == 'lstm' else torch.nn.GRU

        if 'post' in option.feature_extract:
            act_fn_map = dict(relu=torch.nn.ReLU(), tanh=torch.nn.Tanh(), elu=torch.nn.ELU())
            self.base = torch.nn.Sequential(act_fn_map[option.act_fn],
                                            init_(torch.nn.Linear(input_size, option.fc_size)),
                                            act_fn_map[option.act_fn])
            self.recurrent_net = init_rec(base_fn(option.fc_size, option.hidden_size))
        else:
            self.base = torch.nn.ReLU()
            self.recurrent_net = init_rec(base_fn(option.feature_extract['flatten_out'], option.hidden_size))

        self.lstm = option.rec_type == 'lstm'
        self.horizon_length = option.horizon_length

    def attr(self, hxs, fnc):
        return [fnc(hxs[0]), fnc(hxs[1])] if self.lstm else fnc(hxs)

    def forward(self, x, hxs, masks):
        x = self.base(x)

        N = self.attr(hxs, lambda _x: _x.size(0))
        N = N[0] if self.lstm else N
        if x.size(0) == N:
            hxs = self.attr(hxs, lambda _x: (_x * masks).unsqueeze(0))
            x, hxs = self.recurrent_net(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = self.attr(hxs, lambda _x: _x.squeeze(0))
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            T = int(x.size(0) / N)
            N_ = T // self.horizon_length
            N = N * N_
            T = self.horizon_length

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = self.attr(hxs, lambda _x: torch.repeat_interleave(_x.unsqueeze(0), N_, 0).view(N, -1))
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.recurrent_net(
                    x[start_idx:end_idx],
                    self.attr(hxs, lambda _x: _x * masks[start_idx].view(1, -1, 1)))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = self.attr(hxs, lambda _x: _x.squeeze(0))

        return x, hxs


class VisionNet(nn.Module):
    def __init__(self, image_shape, out_size):
        super(VisionNet, self).__init__()
        n, w, h = image_shape
        img_init = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.base = nn.Sequential(
            img_init(nn.Conv2d(n, 32, kernel_size=(8, 8), stride=(4, 4))), nn.ReLU(),
            img_init(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))), nn.ReLU(),
            img_init(nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))), nn.ReLU(), Flatten(),
            img_init(nn.Linear(32 * 7 * 7, out_size)), nn.ReLU())

    def forward(self, inputs):
        return self.base(inputs / 255.0)


class Relational(nn.Module):
    def __init__(self, input_size, heads, n_embed, layer_norm=False, qk_w=1.0, v_w=0.01):
        super().__init__()

        self.n_embed = n_embed
        self.heads = heads

        self._layer_norm = nn.LayerNorm(n_embed) if layer_norm else None

        self._qk = nn.Linear(input_size, n_embed * 2)
        qk_scale = np.sqrt(qk_w / input_size)
        torch.nn.init.xavier_normal_(self._qk.weight, qk_scale)
        torch.nn.init.constant_(self._qk.bias, 0)

        self._v = nn.Linear(input_size, n_embed)
        v_scale = np.sqrt(v_w / input_size)
        torch.nn.init.xavier_normal_(self._v.weight, v_scale)
        torch.nn.init.constant_(self._v.bias, 0)

        self.head_dim = n_embed // heads

    def forward(self, inp):
        bs, NE, features = inp.size()

        if self._layer_norm:
            inp = self._layer_norm(inp)

        qk = self._qk(inp).view(bs, NE, self.heads, self.head_dim, 2)
        q, k = [x.squeeze(-1) for x in torch.unbind(qk, -1)]  # (bs, T, NE, heads, features)
        v = self._v(inp).view(bs, NE, self.heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (bs, heads, NE, n_embd / heads)
        k = k.permute(0, 2, 3, 1)  # (bs, heads, n_embd / heads, NE)
        v = v.permute(0, 2, 1, 3)  # (bs, heads, NE, n_embd / heads)

        return q, k, v


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_size, heads, n_embed,
                 layer_norm=False, qk_w=0.125, v_w=0.125):
        super().__init__()
        self.qkv_embed = Relational(input_size, heads, n_embed, layer_norm, qk_w, v_w)

    def forward(self, inp):
        bs, NE, features = inp.size()
        q, k, v = self.qkv_embed(inp)
        w = torch.softmax(torch.matmul(q, k) / np.sqrt(self.qkv_embed.head_dim), -1)
        scores = torch.matmul(w, v).permute(0, 2, 1, 3)  # (bs, n_output_entities, heads, features)
        n_output_entities = scores.shape[1]
        scores = scores.reshape(bs, n_output_entities, self.qkv_embed.n_embed)
        return scores


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, input_size, heads, n_embed, layer_norm=False, post_sa_layer_norm=False, qk_w=0.125, v_w=0.125,
                 post_w=0.125, max_pool=True):
        super().__init__()
        self.max_pool = max_pool
        self._self_attn = SelfAttentionBlock(input_size, heads, n_embed, layer_norm, qk_w, v_w)

        self._post_attn_mlp = torch.nn.Linear(n_embed, n_embed)
        self._post_norm = torch.nn.LayerNorm(n_embed) if post_sa_layer_norm else None

        post_scale = np.sqrt(post_w / n_embed)
        torch.nn.init.xavier_normal_(self._post_attn_mlp.weight, post_scale)
        torch.nn.init.constant_(self._post_attn_mlp.bias, 0)

    def forward(self, inp):
        x = self._self_attn(inp)
        x = x + inp
        x = self._post_norm(x) if self._post_norm else x
        x = torch.max(x, -2)[0] if self.max_pool else torch.mean(x, -2)
        return x


class ResidualRelational(nn.Module):
    def __init__(self, input_size, heads, n_embed, n_blocks=1, layer_norm=False, post_sa_layer_norm=False, qk_w=0.125,
                 v_w=0.125,
                 post_w=0.125):
        super().__init__()
        self.residual_blocks = nn.Sequential(*tuple([ResidualSelfAttentionBlock(input_size,
                                                                                heads,
                                                                                n_embed,
                                                                                layer_norm,
                                                                                post_sa_layer_norm, qk_w, v_w, post_w)
                                                     for _ in range(n_blocks)]))

    def forward(self, inp):
        return self.residual_blocks(inp)


def build_feature_extract(obs_space: gym.Space, option: PolicyOption, policy: bool):
    if isinstance(obs_space, gym.spaces.Dict):
        spaces = obs_space.spaces
        embed_dict = torch.nn.ModuleDict()
        for k in (option.policy_keys if policy else option.value_keys):
            space = spaces[k]
            if len(space.shape) != 3:
                embed_dict[k] = init_(torch.nn.Linear(int(np.prod(space.shape)), option.feature_extract['embed_out']))
            else:
                embed_dict[k] = VisionNet(space.shape, option.feature_extract['vision_out'])
        return embed_dict, option.feature_extract['embed_out']
    else:
        in_ = np.ravel(obs_space.shape)[0]
        out_size = option.feature_extract['flatten_out']
        model = torch.nn.Sequential( init_(torch.nn.Linear(in_, out_size)),
                                     torch.nn.ReLU(),
                                     init_(torch.nn.Linear(out_size, out_size)))

        return model, out_size