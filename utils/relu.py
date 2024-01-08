import numpy as np

class ReLU:
    
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, grad_output):
        relu_grad = x > 0
        return grad_output * relu_grad