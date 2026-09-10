"""Frozen masked ISAB/PMA regressor for final-snapshot raw halo sets."""
from __future__ import annotations

import torch
from torch import nn

from src.models.deepsets import count_parameters, validate_set_batch

EXPECTED_PARAMETERS = 11_313


def row_ffn() -> nn.Sequential:
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(16, 16), nn.Dropout(0.2))


class MaskedAttentionBlock(nn.Module):
    """Two-head scaled dot-product attention with post-residual normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.q, self.k, self.v, self.out = (nn.Linear(16, 16) for _ in range(4))
        self.attention_dropout = nn.Dropout(0.2)
        self.output_dropout = nn.Dropout(0.2)
        self.norm1, self.norm2 = nn.LayerNorm(16), nn.LayerNorm(16)
        self.ffn = row_ffn()

    def forward(self, queries, keys, key_mask, query_mask=None):
        validate_set_batch(keys, key_mask, 16)
        if query_mask is not None:
            validate_set_batch(queries, query_mask, 16)
        keys = keys.masked_fill(~key_mask.unsqueeze(-1), 0.0)
        def heads(values):
            return values.reshape(values.shape[0], values.shape[1], 2, 8).transpose(1, 2)
        q, k, v = heads(self.q(queries)), heads(self.k(keys)), heads(self.v(keys))
        scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float('-inf'))
        attended = self.attention_dropout(scores.softmax(dim=-1)) @ v
        attended = attended.transpose(1, 2).contiguous().reshape_as(queries)
        values = self.norm1(queries + self.output_dropout(self.out(attended)))
        values = self.norm2(values + self.ffn(values))
        if query_mask is not None:
            values = values.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        return values


class ISAB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(32, 16))
        self.to_inducing = MaskedAttentionBlock()
        self.to_halos = MaskedAttentionBlock()

    def forward(self, x, mask):
        inducing = self.inducing.unsqueeze(0).expand(x.shape[0], -1, -1)
        hidden = self.to_inducing(inducing, x, mask)
        hidden_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=x.device)
        return self.to_halos(x, hidden, hidden_mask, mask)


class PMA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 16))
        # PMA(S, rFF(X)): separate preprocessing FFN, in addition to the MAB FFN.
        self.ffn = row_ffn()
        self.attention = MaskedAttentionBlock()

    def forward(self, x, mask):
        keys = self.ffn(x).masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.attention(self.seed.unsqueeze(0).expand(x.shape[0], -1, -1), keys, mask)


class SetTransformerRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Dropout(0.2), nn.LayerNorm(16))
        self.blocks = nn.ModuleList([ISAB(), ISAB()])
        self.pool = PMA()
        self.regressor = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.2),
                                       nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, ISAB):
                nn.init.xavier_uniform_(module.inducing)
            elif isinstance(module, PMA):
                nn.init.xavier_uniform_(module.seed)
        assert count_parameters(self) == EXPECTED_PARAMETERS

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        validate_set_batch(x, mask)
        x = x.float().masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.projection(x).masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, mask)
        return self.regressor(self.pool(x, mask).squeeze(1))
