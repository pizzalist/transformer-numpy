import numpy as np
from utils import softmax, log_softmax
class ProjectionLayer:
    def __init__(self, d_model: int, vocab_size: int): 
        self.weights = np.random.randn(d_model, vocab_size) * np.sqrt(2. / (d_model + vocab_size))
        self.bias = np.zeros(vocab_size)
        
    def forward(self, x):
        x = np.dot(x, self.weights) + self.bias
        return log_softmax(x)
    
    def __call__(self, x):
        return self.forward(x)