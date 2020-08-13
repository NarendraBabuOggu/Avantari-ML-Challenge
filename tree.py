import os
import glob
from PIL import Image
from random import shuffle
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import collections
import torchvision.transforms as transforms
import torchvision.models as models
from functools import partial
import gc
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
from itertools import combinations
from torchsummary import summary
import time
import pickle as pkl
import torchvision
import pandas as pd
from typing import *
import warnings
from config import *
import argparse
from data import ImageDataset, read_image
try : 
    from annoy import AnnoyIndex
except : 
    print("Annoy is Not Installed. Please installa and run.")
    from annoy import AnnoyIndex
from utils import *
warnings.filterwarnings('ignore')

def main(args:Callable) : 
    features_df = get_pickle_object(args.feature_path)

    search_tree = build_annoy_tree(features_df['Features'], feature_len = 4096, ntree = 500, build = True, save_path = None)

    search_tree.save(args.tree_path)

    neighbors = get_neighbors(features_df, search_tree, args.n_neighbours, args.neighbors_path)


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(prog = 'neighbours.py', description = 'Process The Arguments for neighbours.py.')
    parser.add_argument('--feature_path', type = str, required=True, help = 'Path containing the features of all the Images saved by pickle')
    parser.add_argument('--model_name', type = str, required=True, help = 'Name of the Model')
    parser.add_argument('--tree_path', type = str, required=True, help = 'Path to save the Annoy Tree')
    parser.add_argument('--neighbors_path', type = str, default = 'neighbors.csv', help = 'Path to save the neighbours for all the images')
    parser.add_argument('--n_neighbours', type=int, default = 50, help = 'The number of Neighbours to retrieve')
    
    args = parser.parse_args()

    main(args)