import torch
import torch.nn as nn
import math

# class PositionalEncoding(nn.Module,):
#     def __init__(self, d_model, var_len:bool):
#         super().__init__()
#         self.d_model = d_model
#         self.var_len = var_len

#         if not var_len:
#             ...
            

#     def forward(self, x):
#         # x: [seq_len, batch_size, d_model]
#         seq_len = x.size(0)
#         device = x.device

#         return x + self.sinusoidal_positional_encoding(seq_len, device)

#     def sinusoidal_positional_encoding(self, seq_len, device):
#         position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
#         pe = torch.zeros(seq_len, self.d_model, device=device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(1)  # [seq_len, 1, d_model]
#         return pe    


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.sinusoidal_positional_encoding(max_len, d_model)

#     def sinusoidal_positional_encoding(self, max_len, d_model):
#         pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
#         pe[:, 0::2] = torch.sin(position * div_term)  # even
#         pe[:, 1::2] = torch.cos(position * div_term)  # odd
#         self.pe = pe.unsqueeze(1)  # [max_len, 1, d_model]

#     def forward(self, x):
#         # x: [seq_len, batch_size, d_model]
#         return x + self.pe[:x.size(0)]
    

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.register_buffer("pe", self._generate_pe(max_len, d_model))

    def _generate_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x