import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_curve, auc

def load_training_checkpoint(file_path):
    checkpoint = torch.load(file_path, map_location='cpu')
    return {
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'val_accs': checkpoint['val_accs'],
        'recalls': checkpoint['recalls'],
        'precisions': checkpoint['precisions'],
        'rocs': checkpoint['rocs'],
        'epoch': checkpoint['epoch']
    }

def load_testing_checkpoint(file_path):
    checkpoint = torch.load(file_path, map_location='cpu')
    return {
        'accuracy': checkpoint['accuracy'],
        'recall': checkpoint['recall'],
        'precision': checkpoint['precision'],
        'roc': checkpoint['roc'],
        'f1': checkpoint['f1'],
        'tp': checkpoint['tp'],
        'fp': checkpoint['fp'],
        'tn': checkpoint['tn'],
        'fn': checkpoint['fn'],
        'y_true': checkpoint['y_true'],
        'y_pred': checkpoint['y_pred'],
        'y_scores': checkpoint['y_scores']
    }

def plot_training_metrics(metrics, title=None):
    epochs = list(range(1, len(metrics['train_losses']) + 1))

    fig = plt.figure(figsize=(10, 10))
    if title:
        fig.suptitle(title, fontsize=16)

    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 1. Loss - spans the entire top row
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, metrics['train_losses'], label='Train Loss')
    ax1.plot(epochs, metrics['val_losses'], label='Val Loss')
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylim(0, 0.5)
    ax1.axvline(x=5, color='red', linestyle=':', linewidth=2)
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # 2. Accuracy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, metrics['val_accs'], label='Accuracy', color='green')
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.axvline(x=5, color='red', linestyle=':', linewidth=2)
    ax2.set_ylim(0.5, 1)
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    # 3. Recall
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, metrics['recalls'], label='Recall', color='orange')
    ax3.set_title("Recall")
    ax3.set_xlabel("Epoch")
    ax3.axvline(x=5, color='red', linestyle=':', linewidth=2)
    ax3.set_ylabel("Recall")
    ax3.set_ylim(0.5, 1)
    ax3.grid(True)
    ax3.legend()

    # 4. Precision
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(epochs, metrics['precisions'], label='Precision', color='purple')
    ax4.set_title("Precision")
    ax4.set_xlabel("Epoch")
    ax4.axvline(x=5, color='red', linestyle=':', linewidth=2)
    ax4.set_ylabel("Precision")
    ax4.set_ylim(0.5, 1)
    ax4.grid(True)
    ax4.legend()

    # 5. ROC AUC
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(epochs, metrics['rocs'], label='ROC AUC', color='red')
    ax5.set_title("ROC AUC")
    ax5.set_xlabel("Epoch")
    ax5.axvline(x=5, color='red', linestyle=':', linewidth=2)
    ax5.set_ylabel("ROC AUC")
    ax5.set_ylim(0.5, 1)
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_confusion_matrix(tp, fp, tn, fn, title="Confusion Matrix", labels=["Not Cheating", "Cheating"]):
    # [[TN, FP],
    #  [FN, TP]]
    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 18}  # 🔍 Increase font size here
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def show_metrics(accuracy, precision, recall, f1, specificity=None, auc=None):
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC"],
        "Value": [accuracy, precision, recall, f1, specificity, auc]
    }

    df = pd.DataFrame(data).dropna()

    display(df.style.format({"Value": "{:.4f}"}).hide(axis="index"))

def plot_roc_curve(y_true, y_probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(title, fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.grid(True)
    plt.tight_layout()
    plt.show()