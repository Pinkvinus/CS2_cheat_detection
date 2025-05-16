from training.train import train_model
from training.hyperparameters import feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout
from models.Transformer_v1 import Transformer_V1
import os

model = Transformer_V1(feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout)
project_root = os.path.dirname(os.path.abspath(__file__))

train_model(model, project_root)
