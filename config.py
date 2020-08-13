"""
This file contains the configuration information
"""
import torch
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 64
NUM_WORKERS = 8
SIZE = (224, 224)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'