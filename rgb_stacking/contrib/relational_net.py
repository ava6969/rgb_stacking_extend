import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualSelfAttentionCell(nn.Module):
    def __init__(self, input_shape, heads, embed_dim,
                 layer_norm=True, max_pool_out=None,
                 qk_w=0.125, v_w=0.125, post_w=0.125):
        super().__init__()

        self.n_features, self.feature_size = input_shape
        qk_scale = np.sqrt(qk_w / self.feature_size)
        v_scale = np.sqrt(v_w / self.feature_size)
        self.logit_scale = np.sqrt(embed_dim / heads)
        post_scale = np.sqrt(post_w / embed_dim)

        self.embed_dim = embed_dim
        self.heads = heads
        self.max_pool_out = max_pool_out

        self._qk = nn.Linear(self.feature_size, embed_dim * 2)
        self._v = nn.Linear(self.feature_size, embed_dim)
        self._post_attn_mlp = nn.Linear(embed_dim, self.feature_size)

        self.pre_layer_norm = nn.LayerNorm(self.feature_size, elementwise_affine=True) if layer_norm else None
        self.post_layer_norm = nn.LayerNorm(self.feature_size, elementwise_affine=True) if layer_norm else None

        self._qk.weight = torch.nn.init.xavier_normal_(self._qk.weight, qk_scale)
        self._qk.bias = torch.nn.init.constant_(self._qk.bias, 0)

        self._v.weight = torch.nn.init.xavier_normal_(self._v.weight, v_scale)
        self._v.bias = torch.nn.init.constant_(self._v.bias, 0)

        self._post_attn_mlp.weight = torch.nn.init.xavier_normal_(self._post_attn_mlp.weight, post_scale)
        self._post_attn_mlp.bias = torch.nn.init.constant_(self._post_attn_mlp.bias, 0)

        self.embed_head_ratio = embed_dim // heads

    def _apply_attention(self, inp):
        B, NE, features = inp.size()

        _inp = self.pre_layer_norm(inp) if self.pre_layer_norm else inp

        qk_out = self._qk(_inp).view(B, self.n_features, self.heads, self.embed_head_ratio, 2)

        splitted = qk_out.unbind(-1)
        query = splitted[0].squeeze(-1).permute(0, 2, 1, 3)
        key = splitted[1].squeeze(-1).permute(0, 2, 3, 1)
        val = self._v(inp).view(B, self.n_features, self.heads, self.embed_head_ratio).permute(0, 2, 1, 3)

        return query, key, val

    def forward(self, inp):
        query, key, val = self._apply_attention(inp)
        logit = torch.matmul(query, key) / self.logit_scale
        smax = torch.softmax(logit, -1)

        att_sum = torch.matmul(smax, val).permute(0, 2, 1, 3).reshape(-1, self.n_features, self.embed_dim)
        x = inp + self._post_attn_mlp(att_sum)
        x = self.post_layer_norm(x) if self.post_layer_norm else x

        if self.max_pool_out is not None:
            x = torch.max(x, -2).values if self.max_pool_out else torch.mean(x, -2)

        return x


class ResidualSelfAttention(nn.Module):
    def __init__(self, n_blocks, input_shape, heads, embed_dim,
                 layer_norm=True, max_pool_out=None,
                 qk_w=0.125, v_w=0.125, post_w=0.125):
        super().__init__()
        self.residual_blocks = nn.Sequential(*tuple([ResidualSelfAttentionCell(input_shape,
                                                                               heads=heads,
                                                                               embed_dim=embed_dim,
                                                                               layer_norm=layer_norm,
                                                                               max_pool_out=max_pool_out,
                                                                               qk_w=qk_w, v_w=v_w, post_w=post_w)
                                                     for _ in range(n_blocks)]))

    def forward(self, inp):
        return self.residual_blocks(inp)
