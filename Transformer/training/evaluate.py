import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            preds = probs > 0.5 # Check that this function works

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else None
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = None

    return acc, avg_loss, recall, precision, roc_auc
