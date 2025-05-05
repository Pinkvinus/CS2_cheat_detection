import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from Transformer.utils.dataset import DataImporter
from Transformer.training.evaluate import evaluate
from Transformer.training.scheduler import get_scheduler
# from models.SOMEMODEL import TransformerModel
from Transformer.utils.utils import save_checkpoint
import os


def train_model(batch_size=16, learning_rate=1e-4, num_epochs=10, train_size=0.7, val_size=0.15, test_size=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DataImporter(data_dir="data")
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(optimizer)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(val_acc)
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, f"C:/Users/Gert/repos/AML-exam-project/project/models/autoencoder_Ver11_{epoch+1}_epochs_checkpoint.pth", train_losses, val_losses)

if __name__ == "__main__":
    train_model()
