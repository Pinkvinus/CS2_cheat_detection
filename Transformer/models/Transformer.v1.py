import torch
import torch.nn as nn
import math
from .PositionalEncoding import PositionalEncoding

class TransformerV1(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 1)  # For regression or classification

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        src = src.transpose(0, 1)  # -> [seq_len, batch_size, input_dim]
        src = self.input_proj(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output[-1])  # Use last time step or use mean pooling for classification
        return output  # [batch_size, 1]
