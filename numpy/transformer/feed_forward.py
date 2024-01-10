from ..utils.relu import ReLU
from ..utils.dropout import Dropout
import numpy as np

class FeedForwardBlock:
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        # forward만 구현, random한 값으로 임의 init
        self.W1 = np.random.randn(d_model, d_ff) 
        self.b1 = np.random.randn(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model)
        
        self.actiavation = ReLU()
        self.dropout = Dropout(dropout)
        
    def forward(self, x):
        # linear_1 = self.dropout.forward(self.actiavation.forward(np.dot(x, self.W1) + self.b1))
        # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        linear_1 = np.dot(x, self.W1) + self.b1
        # ReLU
        linear_1 = self.actiavation.forward(linear_1)
        # Dropout
        linear_1 = self.dropout.forward(linear_1)
        
        # (batch, seq_len, d_ff) -->(batch, seq_len, d_model)
        linear_2 = np.dot(linear_1, self.W2) + self.b2
        return linear_2