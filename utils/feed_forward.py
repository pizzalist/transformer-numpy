from .relu import ReLU
from .dropout import Dropout
import numypa as np

class FeedForwardBlock:
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        # forward만 구현, random한 값으로 임의 init
        self.W1 = np.random.randn(d_model, d_ff) 
        self.b1 = np.random.randn(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model)
        
        self.actiavation = ReLU()
        self.dropout = Dropout.forward(dropout)
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        linear_1 = self.dropout(self.actiavation(np.dot(x, self.W1) + self.b1))
        
        # (batch, seq_len, d_ff) -->(batch, seq_len, d_model)
        linear_2 = np.dot(linear_1, self.W2) + self.b2
        return linear_2