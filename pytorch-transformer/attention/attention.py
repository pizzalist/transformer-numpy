import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None: # h = number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # d_model을 h로 나눠질 수 있는지 확인
        assert d_model % h == 0, 'd_model is not divisible by h'
        
        # d_k는 각 attention head의 key, query, value 벡터의 차원
        self.d_k = d_model // h 
        
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v
        self.w_o = nn.Linear(d_model, d_model) # W_o
        
        self.dropout = nn.Dropout(dropout) 

    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        #  query, key, value의 마지막 차원 d_k
        d_k = query.shape[-1] 
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) 
        # attention_scores (batch_size, h, seq_len, seq_len)
        # mask (batch_size, 1, 1, seq_len)
        
        # softmax전 mask씌움
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # mask가 0이면 -1e9
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None: 
            attention_scores = dropout(attention_scores) 
        
        return (attention_scores @ value), attention_scores 
        
    def forward(self, q, k, v, mask): 
        # q, k, v (batch_size, seq_len, d_model)
        query = self.w_q(q) # Q' matrix 
        # query = (8 batch_size, 350 seq_len, 512 d_model)
        key = self.w_k(k) # K' matrix
        value = self.w_v(v) # V' matrix
        
        # 다른 head들에 맞게 더 작은 행렬로 나눔
        # embeddings (3자원)을 h로 나눔 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) 
        # query (batch_size, seq_len, h, d_k) -> (8 batch_size, 8 h, 350 seq_len, 64 d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # x (batch_size 8, h 8 , seq_len 350, d_k 64), attention_scores (batch_size, h, seq_len, seq_len)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # x (8 batch_size, 350 seq_len, 512 d_model)
        
        
        return self.w_o(x)
    
    