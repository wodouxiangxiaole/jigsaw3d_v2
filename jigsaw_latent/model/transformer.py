import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0:1, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e4)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = th.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        
        output = F.scaled_dot_product_attention(q, k, v, mask.unsqueeze(1))

        # output = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.gen_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_model * 2, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, self_mask, gen_mask):
        # assert (gen_mask.max() == 1 and gen_mask.min() == 0), f"{gen_mask.max()}, {gen_mask.min()}"
        x2 = self.norm_1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
                + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x
