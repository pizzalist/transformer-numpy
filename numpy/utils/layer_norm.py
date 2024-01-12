import numpy as np

class LayerNormalization:
    
    def __init__(self, eps: float = 10**-6): # We define epsilon as 0.000001 to avoid division by zero
        self.eps = eps
        
        # 학습은 구현 안함
        self.alpha = np.ones((1,)) 
        self.bias = np.zeros((1,))
        
    def forward(self, x):
        mean = x.mean(axis = -1, keepdims = True) # Computing the mean of the input data. Keeping the number of dimensions unchanged
        std = x.std(axis = -1, keepdims = True) # Computing the standard deviation of the input data. Keeping the number of dimensions unchanged
        
        # Returning the normalized input
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
    def __call__(self, x):
        return self.forward(x)