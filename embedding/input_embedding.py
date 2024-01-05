import numpy as np
import math


# Creating Input Embeddings
class InputEmbeddings:
    
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model # Dimension of vectors (512)
        self.vocab_size = vocab_size # Size of the vocabulary
        self.embedding = np.random.randn(vocab_size, d_model) # PyTorch layer that converts integer indices to dense embeddings
        
        # nn.Embedding은 이니셜한 값에서 학습하는 과정이 있는데 numpy에서는 구현안되나??
        
    def forward(self, x):
        return self.embedding[x] * math.sqrt(self.d_model) # Normalizing the variance of the embeddings