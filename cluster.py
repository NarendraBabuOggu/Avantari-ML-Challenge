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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


def main(args:Callable) : 
    tsne = TSNE(n_components=2, random_state=1997)

    features_df = get_pickle_object(args.feature_path)

    features = np.asarray(features_df['Features'].tolist())
    logging.info("Applying TSNE")
    reduced_features = tsne.fit_transform(features)

    save_pickle_object(reduced_features, args.tsne_path)
    save_pickle_object(reduced_features, args.tsne_feature_path)

    plt.scatter(x = reduced_features[:,0], y=reduced_features[:,1])

    plt.savefig('tsne_plot.jpg')

    logging.info("Applying KMeans")
    kmeans = KMeans(algorithm='auto', random_state = 1997, max_iter = 500, n_clusters = 6)
    logging.info(kmeans)
    labels = kmeans.fit_predict(reduced_features)

    save_pickle_object(kmeans, args.kmeans_path)

    features_with_cluster = features_df.copy()

    features_with_cluster['cluster'] = labels

    logging.info("Saving Image CLusters")
    features_with_cluster[['ImageName', 'cluster']].to_csv(args.image_cluster_path, index = False)

    plot_clusters(reduced_features, labels, kmeans, 'K-means clustering on the dataset (TSNE-reduced data)\n'
          'Centroids are marked with white cross', plot_centroids = True)
    logging.info("CLuster Plot is save at cluster_plot.jpg")

if __name__ == '__main__' : 

    parser = argparse.ArgumentParser(prog = 'neighbours.py', description = 'Process The Arguments for neighbours.py.')
    parser.add_argument('--feature_path', type = str, required=True, help = 'Path containing the features of all the Images saved by Pickle')
    parser.add_argument('--kmeans_path', type = str, required=True, help = 'Path to save the kmeans model by pickle')
    parser.add_argument('--image_cluster_path', type = str, required=True, help = 'Path to save the image and cluster details as csv')
    parser.add_argument('--tsne_path', type = str, required=True, help = 'Path to save TSNE Model by Pickle')
    parser.add_argument('--tsne_feature_path', type = str, required=True, help = 'Path to save the TSNE Reuced Features by Pickle')
    
    args = parser.parse_args()

    main(args)