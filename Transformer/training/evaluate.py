import torch
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze(1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            preds = outputs > 0.5 # Check that this function works

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else None
    acc = accuracy_score(all_labels, all_preds)

    return acc, avg_loss
