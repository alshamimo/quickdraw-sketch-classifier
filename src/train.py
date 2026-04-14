"""
Training loop for neural network models.

Manages the training process including forward/backward passes,
loss calculation, optimization, and tracking metrics per epoch.
Runs validation after each training epoch to monitor progress.
"""
import torch
from torch import nn


class Train:
    """
    Handles model training with validation after each epoch.

    Tracks training and validation metrics (loss, accuracy) in history dict
    for later analysis and visualization.
    """

    def __init__(self, model, data_loader, test_loader, epochs=20, lr=0.001, device="cpu"):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr)
        self.history = {
            'train_loss': [],
            'train_acc':  [],
            'val_loss': [],
            'val_acc': []
        }
        self.test_loader = test_loader
        self.device = device

    def train(self, eval_func):
        """
        Execute training loop for all epochs.

        Runs training phase (with gradients), then validation phase (no gradients)
        after each epoch. Logs progress to console.

        Args:
            eval_func: Function to evaluate model, returns (val_loss, val_acc)

        Returns:
            history: Dict containing training/validation metrics per epoch
        """
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.data_loader:
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = train_loss / len(self.data_loader)
            epoch_acc  = correct / total
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)

            print(f"Epoch {epoch+1:02d}/{self.epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:.4f}")
            val_loss, val_acc = eval_func(self.model, self.test_loader, self.device)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

        return self.history