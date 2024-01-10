import numpy as np

from utils import softmax

class Attention:
    def forward(self, query, key, value, mask=None, dropout=None):
        # query, key, value shape: (batch_size, seq_len, d_k)
        # mask shape: (batch_size, seq_len, seq_len)
        scores = np.matmul(query, key.transpose(0,1,3,2)) / np.sqrt(query.shape[-1])
        
        # make가 0인 부분은 -1e9로 채워준다.
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        # softmax 적용
        p_attn = softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout.forward(p_attn)
            
        return np.matmul(p_attn, value), p_attn