import torch.nn as nn
from .layer_norm import LayerNormalization
from .dropout import Dropout

class ResidualConnection:
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self,dropout):
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout.forward(sublayer(self.norm.forward(x)))
    
    def __call__(self, x, sublayer):
        return self.forward(x, sublayer)
