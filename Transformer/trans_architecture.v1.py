import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self,  d_model:int, vocab_size:int):
        """
            d_model    : the dimensions of the model 
            vocab_size : how many words there are in the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size


    def sinusoidal_positional_encoding(seq_len, d_model, device='cpu'):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.to(device)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
            d_model: the size of the vector that the sequence should be
            seq_len: The maximum length of the input
            dropout: Ensures less overfitting
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

    def 