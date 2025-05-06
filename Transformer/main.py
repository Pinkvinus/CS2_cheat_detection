from training.train import train_model
from training.hyperparameters import batch_size, learning_rate, num_epochs, train_size, val_size, test_size, feature_dim, seq_len, dimension_model, number_heads, num_layers, dim_feedforward, dropout
from models.Transformer_v1 import Transformer_V1

model = Transformer_V1(feature_dim, seq_len, dimension_model, number_heads, num_layers, dim_feedforward, dropout)

train_model(model)
