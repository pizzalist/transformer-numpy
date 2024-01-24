import torch
import torch.nn as nn
from attention import MultiHeadAttentionBlock
from transformer.feed_forward import FeedForwardBlock
from utils import ResidualConnection, LayerNormalization

class EncoderBlock(nn.Module):
    
    # This block takes in the MultiHeadAttentionBlock and FeedForwardBlock, as well as the dropout rate for the residual connections
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # 2 Residual connections with dropout
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) 
        
    def forward(self, x, src_mask):
        # MHA -> ADD&NORM 
        # query, key, and value에 x 입력
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) 
        
        # FFN -> ADD&NORM
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    # The Encoder takes in instances of 'EncoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        # EncoderBlocks 에서 저장
        self.layers = layers 
        # encoder의 output을 normalize
        self.norm = LayerNormalization() 
        
    def forward(self, x, mask):
        # self.layers에 저장된 각각의 EncoderBlock
        for layer in self.layers:
            x = layer(x, mask) 
        return self.norm(x) # Normalizing output