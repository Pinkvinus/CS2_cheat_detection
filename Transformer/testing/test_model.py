import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from models.Transformer_v1 import Transformer_V1
from data.dataset import DataImporter
from pathlib import Path
import os
from training.hyperparameters import feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout, test_size, train_size, val_size, batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise Exception("Cuda could not be activated")

root_folder = Path(__file__).parent.parent
checkpoint_name = "model_1024_epoch_99.pth"

model = Transformer_V1(feature_dim, seq_len, number_heads, num_layers, dim_feedforward, dropout)
checkpoint = torch.load(os.path.join(root_folder, "checkpoints", checkpoint_name), map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

dataset = DataImporter()
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

y_true = all_labels
y_pred = all_preds
y_scores = all_probs

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_scores)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc:.4f}")

data = {
    'accuracy': acc,
    'recall': recall,
    'precision': precision,
    'roc': roc,
    'f1': f1,
    'tp': int(tp),
    'fp': int(fp),
    'tn': int(tn),
    'fn': int(fn),
    'y_true': y_true,
    'y_pred': y_pred,
    'y_scores': y_scores
}
file_path_save = os.path.join(root_folder, "checkpoints", checkpoint_name.replace(".pth", "_testdata.pth"))
torch.save(data, file_path_save)

print(f"file saved to: {file_path_save}")