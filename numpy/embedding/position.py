import numpy as np
from utils import Dropout

class PositionalEncoding:
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Dimensionality of the model
        self.seq_len = seq_len # Maximum sequence length
        self.dropout = Dropout(dropout) # Dropout layer to prevent overfitting
        
        # Creating a positional encoding matrix of shape (seq_len, d_model) filled with zeros
        pe = np.zeros((seq_len, d_model)) 
        
        # Creating a tensor representing positions (0 to seq_len - 1)
        position = np.arange(seq_len, dtype=float)[:, np.newaxis] # Transforming 'position' into a 2D tensor['seq_len, 1']
        
        # Creating the division term for the positional encoding formula
        div_term = np.exp(np.arange(0, d_model, 2, dtype=float) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices in pe
        # pe = (350, 512)
    
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = np.expand_dims(pe, axis=0)
        
        self.pe = pe
        
    def forward(self,x):
        # Addind positional encoding to the input tensor X
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout.forward(x) # Dropout for regularization
    
    def __call__(self, x):
        return self.forward(x)