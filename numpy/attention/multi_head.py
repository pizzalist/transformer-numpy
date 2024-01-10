import numpy as np

from .single import Attention
from utils import Dropout

class MultiHeadedAttention:
    def __init__(self, h, d_model, dropout=0.1):
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        
        self.linear_layers = [np.random.randn(d_model, d_model) for _ in range(3)]
        self.output_linear = np.random.randn(d_model, d_model)
        self.attention = Attention()
        
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
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h*d_k = d_model)
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.h * self.d_k)
        print(x.shape)
        return np.matmul(x, self.output_linear)