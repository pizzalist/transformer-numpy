import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None: # epsilon 0.000001 으로 지정하여 0으로 나누는 것을 방지 
        super().__init__()
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones(1)) 
        
        self.bias = nn.Parameter(torch.zeros(1)) 
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True) 
        
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
