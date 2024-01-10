import torch.nn as nn
import torch.nn.functional as F
import torch

import math

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    # query, key, value: (batch_size, seq_len, d_k)
    
    # query, key, value: (batch_size, h, seq_len, d_k)
    
    # mask: (batch_size, h, seq_len, seq_len)
    
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1) # (batch_size, h,  seq_len, seq_len)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
