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
import argparse
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
try : 
    from annoy import AnnoyIndex
except : 
    print("Annoy is Not Installed. Please installa and run.")
    from annoy import AnnoyIndex
from utils import *
warnings.filterwarnings('ignore')

def main(args:Callable) : 
    img_paths = get_file_names(args.images_path)
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(img_paths, tfs, size = SIZE)
    dataloader = DataLoader(dataset, shuffle = True, num_workers = NUM_WORKERS, 
                              batch_size=BATCH_SIZE)
    model = get_model(args.model_name)
    
    logging.info(summary(model, (3, 224, 224)))

    sf = SaveFeatures(list(model.children())[-1][4])

    paths, features = get_features(dataloader, sf, model)

    features_df = pd.DataFrame([paths, features]).T
    features_df['img_path'] = features_df[0].apply(lambda x: x.split('/')[-1])
    features_df.columns = ['FullPath', 'Features', 'ImageName']

    save_pickle_object(features_df, args.save_feature_path)




if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(prog = 'neighbours.py', description = 'Process The Arguments for neighbours.py.')
    parser.add_argument('--images_path', type = str, required=True, help = 'Path of Image')
    parser.add_argument('--model_name', type = str, required=True, help = 'Name of the Model')
    parser.add_argument('--save_feature_path', type = str, required=True, help = 'Path to save the Image Features')
    args = parser.parse_args()

    main(args)