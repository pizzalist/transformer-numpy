from attention import MultiHeadAttentionBlock
from transformer.feed_forward import FeedForwardBlock
from utils import ResidualConnection, LayerNormalization

class EncoderBlock:
    # This block takes in the MultiHeadAttentionBlock and FeedForwardBlock, as well as the dropout rate for the residual connections
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # Storing the self-attention block and feed-forward block
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(dropout), ResidualConnection(dropout)]
        
    def forward(self, x, src_mask):
        # Applying the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Three 'x's corresponding to query, key, and value inputs plus source mask
        
        # Applying the second residual connection with the feed-forward block 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x # Output tensor after applying self-attention and feed-forward layers with residual connections.
    
    def __call__(self, x, mask):
        return self.forward(x, mask)

class Encoder:
    # The Encoder takes in instances of 'EncoderBlock'
    def __init__(self, layers):
        self.layers = layers # Storing the EncoderBlocks
        self.norm = LayerNormalization() # Layer for the normalization of the output of the encoder layers
        
    def forward(self, x, mask):
        # Iterating over each EncoderBlock stored in self.layers
        for layer in self.layers:
            x = layer(x, mask) # Applying each EncoderBlock to the input tensor 'x'
        return self.norm(x) # Normalizing output
    
    def __call__(self, x, mask):
        return self.forward(x, mask)
