import torch
import torch.nn as nn
from rgb_stacking.contrib.arguments import PolicyOption
from rgb_stacking.contrib.common import init_, init_rec


class RecurrentNet(nn.Module):
    def __init__(self, input_size, action_sz, option: PolicyOption):
        super(RecurrentNet, self).__init__()

        if 'post' in option.feature_extract:
            act_fn_map = dict(relu=torch.nn.ReLU(), tanh=torch.nn.Tanh(), elu=torch.nn.ELU())
            self.base = torch.nn.Sequential(act_fn_map[option.act_fn],
                                            init_(torch.nn.Linear(input_size, option.fc_size)),
                                            act_fn_map[option.act_fn])
            self.recurrent_net = init_rec(torch.nn.LSTM(option.fc_size, option.hidden_size))
        else:
            self.base = torch.nn.ReLU()
            self.recurrent_net = init_rec(torch.nn.LSTM(option.feature_extract['flatten_out'] + action_sz,
                                                        option.hidden_size))

        self.horizon_length = option.horizon_length

    def forward(self, x, hxs, masks):
        x = self.base(x)

        N = hxs[0].size(0)

        if x.size(0) == N:
            hxs = [(s * masks).unsqueeze(0) for s in hxs]
            x, hxs = self.recurrent_net(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = [s.squeeze(0) for s in hxs]
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
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            has_zeros = [has_zeros.item() + 1] if has_zeros.dim() == 0 else (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = [torch.repeat_interleave(s.unsqueeze(0), N_, 0).view(N, -1) for s in hxs]
            outputs = []

            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.recurrent_net(x[start_idx:end_idx],
                                                     [s * masks[start_idx].view(1, -1, 1) for s in hxs])

                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = [s.squeeze(0) for s in hxs]

        return x, hxs
