import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, x, train_flag=True):
        # 학습 시
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)
    
    def backward(self, dout):
        return dout * self.mask