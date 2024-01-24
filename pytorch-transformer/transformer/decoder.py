import torch
import torch.nn as nn
from attention import MultiHeadAttentionBlock
from transformer.feed_forward import FeedForwardBlock
from utils import ResidualConnection, LayerNormalization

class DecoderBlock(nn.Module):
    
    # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention.
    # It also takes in the feed-forward block and the dropout rate
    def __init__(self,  self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # List of three Residual Connections with dropout rate
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) #(8, 350, 512)
        
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) #(8, 350, 512)
        
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization() 
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) 