import numpy as np
from utils import softmax, Dropout

class MultiHeadAttentionBlock:
    def __init__(self, d_model, h, dropout=0.1):
        
        self.d_k = d_model // h
        self.h = h
        
        self.linear_layers = [np.random.randn(d_model, d_model) for _ in range(3)]
        self.output_linear = np.random.randn(d_model, d_model)
        
        self.dropout = Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_embed)
        # mask: (batch_size, seq_len, seq_len)
        
        batch_size = query.shape[0]
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value: (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query, key, value = [np.matmul(x, l).reshape(batch_size, -1, self.h, self.d_k).transpose(0, 2, 1, 3) \
                                for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: (batch_size, h, seq_len, d_k)
        x, attn = MultiHeadAttentionBlock.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h*d_k = d_model)
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.h * self.d_k)
        return np.matmul(x, self.output_linear)
    
    def __call__ (self, query, key, value, mask=None):
        return self.forward(query, key, value, mask)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        # query, key, value shape: (batch_size, seq_len, d_k)
        # mask shape: (batch_size, seq_len, seq_len)
        attention_scores = np.matmul(query, key.transpose(0,1,3,2)) / np.sqrt(query.shape[-1])
        
        # make가 0인 부분은 -1e9로 채워준다.
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)
        # softmax 적용
        attention_scores = softmax(attention_scores, dim=-1)
        
        if dropout is not None:
            attention_scores = dropout.forward(attention_scores)
            
        return np.matmul(attention_scores, value), attention_scores