import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import Preprocessor
import config
import pytest
import torch 

@pytest.fixture(scope="module")
def data_loaders(): 
    p = Preprocessor(config.CLASSES, config.MAX_SAMPLES)
    train_loader, test_loader = p.get_loaders(
        config.TEST_SPLIT,
        config.RANDOM_SEED,
        config.BATCH_SIZE
    )
    return train_loader, test_loader 

def test_images_labels(data_loaders):
    train_loader, test_loader = data_loaders
    images, labels = next(iter(train_loader))
    
    assert images.shape == torch.Size([32, 1, 28, 28])
    assert labels.shape == torch.Size([32])
   
    assert len(labels[:8].tolist()) == 8 
    
def test_train_test_loader(data_loaders):
    train_loader, test_loader = data_loaders
    assert len(train_loader) == 750
    assert len(test_loader) == 188