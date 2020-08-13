import os
import glob
import argparse
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
from data import ImageDataset, read_image
import logging
try : 
    from annoy import AnnoyIndex
except : 
    print("Annoy is Not Installed. Please installa and run.")
    from annoy import AnnoyIndex
from utils import *
warnings.filterwarnings('ignore')


def main(args : Callable)  : 
    """
    Takes the Arguments and Saves the Neighbours for the given Image
    """ 

    features = get_pickle_object(args.feature_path)
    model = get_model('vgg') # Hardcoding the Model Name for now
    logging.info("Building Annoy Tree")
    annoy_tree = build_annoy_tree(
        features, feature_len = 4096, ntree = 500,build = False, save_path = args.tree_path
    )
    sf = SaveFeatures(list(model.children())[-1][4])
    logging.info("Getting Neighbors")
    neighbours = get_similar_images(
        args.image_path, features, sf, annoy_tree, model, args.n_neighbours
    )

    if args.save_neighbours == 1 : 
        save_image_grid(args.image_path, neighbours, path = args.neighbours_path)



if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(prog = 'neighbours.py', description = 'Process The Arguments for neighbours.py.')
    parser.add_argument('--image_path', type = str, required=True, help = 'Path of Image')
    parser.add_argument('--feature_path', type = str, required=True, help = 'Path containing the features of all the Images saved by pickle')
    parser.add_argument('--tree_path', type = str, required=True, help = 'Path containing the Built Annoy Tree saved by pickle')
    parser.add_argument('--model_name', type = str, required=True, help = 'Name of the Model')
    parser.add_argument('--save_neighbours', type = int, default = 0, help = 'To save the Neighbours as a Image or not - use 1 for Yes and 0 for No')
    parser.add_argument('--neighbours_path', type = str, default = 'grid.jpg', help = 'Path to save the Neighbours as a Image')
    parser.add_argument('--n_neighbours', type=int, default = 50, help = 'The number of Neighbours to retrieve')
    args = parser.parse_args()

    main(args)