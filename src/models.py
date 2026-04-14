"""
Neural network models for QuickDraw classification.

Input: 28x28 grayscale images
Output: 5 class predictions (apple, star, fork, candle, eyeglasses)

Two architectures are provided:
- QuickDrawNN: Simple fully-connected baseline
- QuickDrawCNN: Convolutional network for better spatial feature extraction
"""
import torch
from torch import nn
import torch.nn.functional as F


class QuickDrawNN(nn.Module):
    """
    Fully-connected neural network baseline.

    Architecture: Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 5)
    Suitable for comparison against CNN performance on image data.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


class QuickDrawCNN(nn.Module):
    """
    Convolutional neural network for image classification.

    Architecture:
    - Conv2D(1, 16) -> ReLU -> MaxPool
    - Conv2D(16, 32) -> ReLU -> MaxPool
    - Flatten -> Linear(32*5*5, 64) -> ReLU -> Linear(64, 5)

    Better at capturing spatial patterns like edges and shapes compared to NN.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=3)
        
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))

        return self.fc2(x)
