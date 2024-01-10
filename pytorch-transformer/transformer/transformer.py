import torch
import torch.nn as nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.projection_layer import ProjectionLayer
from embedding import InputEmbeddings, PositionalEncoding

class Transformer(nn.Module):
    
    # This takes in the encoder and decoder, as well the embeddings for the source and target language.
    # It also takes in the Positional Encoding for the source and target language, as well as the projection layer
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    # Encoder     
    def encode(self, src, src_mask):
        # (8 batch_size, 350 seq_len)
        src = self.src_embed(src) # Applying source embeddings to the input source language
        # (8 batch_size, 350 seq_len, 512 d_model)
        
        src = self.src_pos(src) # Applying source positional encoding to the source embeddings
        # (8 batch_size, 350 seq_len, 512 d_model)
        return self.encoder(src, src_mask) # Returning the source embeddings plus a source mask to prevent attention to certain elements
    
    # Decoder
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings
        
        # Returning the target embeddings, the output of the encoder, and both source and target masks
        # The target mask ensures that the model won't 'see' future elements of the sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # Applying Projection Layer with the Softmax function to the Decoder output
    def project(self, x):
        return self.projection_layer(x)
