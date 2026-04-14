"""
Model evaluation on validation/test set.

Provides standardized evaluation metrics (loss and accuracy) for
comparing model performance during and after training.
"""
import torch
from torch import nn


def evaluate(model, test_loader, device):
    """
    Calculate loss and accuracy on test data.

    Runs inference on all test batches without gradient computation
    to determine model performance on held-out data.

    Args:
        model: Trained neural network model
        test_loader: DataLoader with test/validation data
        device: Device to run evaluation on ('cpu' or 'cuda')

    Returns:
        val_loss: Average cross-entropy loss across all batches
        val_acc: Classification accuracy (correct predictions / total samples)
    """
    model.eval()
    val_loss = 0
    val_acc = 0
    total = 0
    val_correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / total
        val_loss = val_loss / len(test_loader)
        print(f"Validation loss: {val_loss} | "
                f"Validation accuracy: {val_acc}")

    return val_loss, val_acc
