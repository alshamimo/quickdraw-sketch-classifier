import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import Preprocessor
import config
import pytest

# Einmal laden — für alle Tests
@pytest.fixture(scope="module")
def data():
    p = Preprocessor(config.CLASSES, config.MAX_SAMPLES)
    X, y = p.load_and_preprocess()
    return X, y

def test_output_shapes(data):
    X, y = data
    assert X.shape == (30000, 1, 28, 28)
    assert y.shape == (30000,)

def test_normalization(data):
    X, y = data
    assert X.min() >= 0.0
    assert X.max() <= 1.0

def test_class_balance(data):
    X, y = data
    for i in range(len(config.CLASSES)):
        count = (y == i).sum()
        assert count == config.MAX_SAMPLES