import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES     = ['apple', 'star', 'fork', 'candle', 'eyeglasses']
MAX_SAMPLES = 6000       # Per class
TEST_SPLIT  = 0.2        # 80% Train, 20% Test
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 0.001
RANDOM_SEED = 42         # reproducibility