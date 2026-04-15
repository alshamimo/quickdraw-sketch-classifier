"""
Unit tests for QuickDraw neural network models.

Verifies model architecture requirements:
- Input/output tensor shapes match expected dimensions
- Model parameter counts are within expected ranges (sanity check)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from src.models import QuickDrawNN, QuickDrawCNN

@pytest.fixture(scope="module")
def data():
    x = torch.randn(32, 1, 28, 28)  
    nn_model  = QuickDrawNN()
    cnn_model = QuickDrawCNN()
    return x, nn_model, cnn_model  

def test_output_shapes(data):
    x, nn_model, cnn_model = data 
    nn_out = nn_model(x)
    cnn_out = cnn_model(x)
    
    assert nn_out.shape == torch.Size([32, 5])
    assert cnn_out.shape == torch.Size([32, 5])

def test_models_parameters(data):
    x, nn_model, cnn_model = data 
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert count_params(nn_model) == 109_061
    assert count_params(cnn_model) == 56_389