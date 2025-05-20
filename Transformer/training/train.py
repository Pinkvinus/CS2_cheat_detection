import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data.dataset import DataImporter
from training.evaluate import evaluate
from training.scheduler import get_scheduler
from utils.utils import save_checkpoint
from torchinfo import summary
from training.hyperparameters import seq_len, batch_size, num_epochs, learning_rate, train_size, val_size, test_size, feature_dim, checkpoint_freq
import os


def train_model(model, project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Exception("CUDA Could not be activated!")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = DataImporter()
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(optimizer, step_size=10, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    summary(model=model, input_size=(batch_size, seq_len, feature_dim), device=device.type)

    train_losses = []
    val_losses = []
    val_accs = []
    recalls = []
    precisions = []
    rocs = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        val_acc, val_loss, recall, precs, roc = evaluate(model, val_loader, device, criterion)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        recalls.append(recall)
        precisions.append(precs)
        rocs.append(roc)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        #if (epoch+1) % checkpoint_freq == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, train_losses, val_losses, val_accs, recalls, precisions, rocs)
