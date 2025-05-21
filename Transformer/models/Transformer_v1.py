import torch
import torch.nn as nn
import math
from .PositionalEncoding import PositionalEncoding

class Transformer_V1(nn.Module):
    def __init__(
        self,
        feature_dim=44,         # nr of features per tick
        seq_len=1024,           # nr of ticks
        #d_model=128,            # model dimension (embedding size), NOT IN USE
        nhead=8,                # nr of attention heads
        num_layers=4,           # nr of transformer encoder layers
        dim_feedforward=512,    # hidden size of feedforward network (MLP)
        dropout=0.1             # dropout rate
    ):
        super(Transformer_V1, self).__init__()

        # self.input_proj = nn.Linear(feature_dim, d_model)  # project input to model dimension, basically creating an "embedding" (similar to LLMs)

        self.positional_encoding = PositionalEncoding(d_model=feature_dim, max_len=seq_len + 1)  # +1 for CLS token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )   # input shape is (batch, seq_len, d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))  # add classification token

        self.fc_out = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim) = (batch_size, 1024, 44)
        B = x.size(0)

        # x = self.input_proj(x)  # (batch_size, 1024, d_model) NOT IN USE

        # add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # -> (batch_size, 1025, d_model)

        x = self.positional_encoding(x) # add positional encoding

        x = self.transformer_encoder(x)  # -> (batch_size, 1025, d_model)

        cls_output = x[:, 0]  # get output for classification token
        out = self.fc_out(cls_output)  # -> (batch_size, 1)

        return out.squeeze(1)  # -> (batch_size,)