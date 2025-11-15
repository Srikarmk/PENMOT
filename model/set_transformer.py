import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)

        return out, attn


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key_value, mask=None):
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value, mask)
        query = self.norm1(query + self.dropout1(attn_out))

        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query, attn_weights


class SetTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=3, d_ff=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.self_attn_blocks = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        print(f"[OK] SetTransformer: {d_model}D, {num_heads} heads, {num_layers} layers")

    def forward(self, query_set, key_set, query_mask=None, key_mask=None):
        query_set = query_set.unsqueeze(0) if query_set.dim() == 2 else query_set
        key_set = key_set.unsqueeze(0) if key_set.dim() == 2 else key_set

        for self_attn_block in self.self_attn_blocks:
            query_set = self_attn_block(query_set, query_mask)

        for cross_attn_block in self.cross_attn_blocks:
            query_set, _ = cross_attn_block(query_set, key_set, key_mask)

        return query_set.squeeze(0) if query_set.size(0) == 1 else query_set