import torch
import torch.nn as nn
from rgb_stacking.contrib.arguments import PolicyOption
from rgb_stacking.contrib.common import init_, init_rec


class RecurrentNet(nn.Module):
    def __init__(self, input_size, option: PolicyOption):
        super(RecurrentNet, self).__init__()
        base_fn = torch.nn.LSTM if option.rec_type == 'lstm' else torch.nn.GRU
        ACTION_EMBED_REWARD = 10*4 + 2  # [4 -> 10] + [1 -> gripper action] + [1 -> reward]

        if 'post' in option.feature_extract:
            act_fn_map = dict(relu=torch.nn.ReLU(), tanh=torch.nn.Tanh(), elu=torch.nn.ELU())
            self.base = torch.nn.Sequential(act_fn_map[option.act_fn],
                                            init_(torch.nn.Linear(input_size, option.fc_size)),
                                            act_fn_map[option.act_fn])
            self.recurrent_net = init_rec(base_fn(option.fc_size + ACTION_EMBED_REWARD, option.hidden_size))
        else:
            self.base = torch.nn.ReLU()
            self.recurrent_net = init_rec(base_fn(option.feature_extract['flatten_out'] + ACTION_EMBED_REWARD,
                                                  option.hidden_size))

        self.lstm = option.rec_type == 'lstm'
        self.horizon_length = option.horizon_length

    def attr(self, hxs, fnc):
        return [fnc(hxs[0]), fnc(hxs[1])] if self.lstm else fnc(hxs)

    def forward(self, x, hxs, masks):
        x = torch.concat([self.base(x["input"]), x['past_action'], x['past_reward']], -1)

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

