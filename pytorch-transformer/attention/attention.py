import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None: # h = number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # We ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % h == 0, 'd_model is not divisible by h'
        
        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = d_model // h # d_k formula, like in the original "Attention Is All You Need" paper
        
        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v
        self.w_o = nn.Linear(d_model, d_model) # W_o
        
        self.dropout = nn.Dropout(dropout) # Dropout layer to avoid overfitting
        
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):# mask => When we want certain words to NOT interact with others, we "hide" them
        
        d_k = query.shape[-1] # The last dimension of query, key, and value
        
        # We calculate the Attention(Q,K,V) as in the formula in the image above 
        
        # attention_scores를 모두 더하면 1인 확률값임
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # @ = Matrix multiplication sign in PyTorch
        # attention_scores (batch_size, h, seq_len, seq_len)
        
        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None: # If a mask IS defined...
            attention_scores.masked_fill_(mask == 0, -1e9) # Replace each value where mask is equal to 0 by -1e9
        attention_scores = attention_scores.softmax(dim = -1) # Applying softmax
        if dropout is not None: # If a dropout IS defined...
            attention_scores = dropout(attention_scores) # We apply dropout to prevent overfitting
            
        # 왜 attention_scores @ value를 하는지 모르겠음 
        return (attention_scores @ value), attention_scores # Multiply the output matrix by the V matrix, as in the formula
        
    def forward(self, q, k, v, mask): 
        # q, k, v (batch_size, seq_len, d_model)
        query = self.w_q(q) # Q' matrix 
        # query = (8 batch_size, 350 seq_len, 512 d_model)
        key = self.w_k(k) # K' matrix
        value = self.w_v(v) # V' matrix
        
        
        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        # query (batch_size, h, seq_len, d_k) -> (8 batch_size, 8 h, 350 seq_len, 64 d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        # key (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        # value = (8 batch_size, 8 h, 350 seq_len, 64 d_k)
        
        # Obtaining the output and the attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # x (batch_size, seq_len, d_model), attention_scores (batch_size, h, seq_len, seq_len)
        
        # Obtaining the H matrix
        # x (8 batch_size, 350 seq_len, 512 d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        
        return self.w_o(x) # Multiply the H matrix by the weight matrix W_o, resulting in the MH-A matrix