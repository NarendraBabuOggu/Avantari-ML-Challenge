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
from tqdm import tqdm
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
import logging
from data import ImageDataset, read_image
try : 
    from annoy import AnnoyIndex
except : 
    print("Annoy is Not Installed. Please installa and run.")
    from annoy import AnnoyIndex
warnings.filterwarnings('ignore')

def get_file_names(path:str) -> List[str] : 
    """
    To get all the file names from given pattern using glob

    params : 
        path - string - contains the path pattern

    returns : 
        A List containing all the paths that satisfies the given pattern
    """ 
    img_paths = glob.glob(path)
    shuffle(img_paths)
    return img_paths

def get_features(dataloader:DataLoader, model:Callable, sf:Callable) : 
    """
    To Extract the Features from a DataLoader

    params : 
        dataloader      :   PyTorch Dataloader - contains the images
        model           :   Neural Network Model - to get features
        sf              :   Save Features Class to get features

    returns : 
        returns the Features for all Images as a Numpy Array
    """
    paths = []
    with torch.no_grad() : 
        for batch in tqdm(dataloader) : 
            model(batch['image'].to(DEVICE))
            paths.extend(batch['image_path'])
        
    features = sf.features
    return features

def get_dataloader(image_paths: List[str], transform: Collection[Callable] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None, shuffle:bool = True, 
                 num_workers:int = NUM_WORKERS, batch_size:int = BATCH_SIZE) -> DataLoader : 
    """
    To get the dataloader for ImageDataset using the given image paths

    params : 
        image_paths     :   List of strings - contains the path of Images
        transforms      :   torchvision.transforms.transform - contains one 
                            or more transformations to apply on the images
        size            :   Tuple - contains the height and width or integer 
                            indicating same height and width
        shuffle         :   Boolean - To shuffle the data or not
        num_workers     :   Int - The number of workers to use 
        batch_size      :   Int - Batch Size for the dataloader(number of images to load as a batch)
    """
    dataset = ImageDataset(image_paths, transform, size)
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers = num_workers)
    return dataloader

def get_model(model_name : str = 'vgg', train:bool = False) -> nn.Module : 
    """
    Returns the PyTorch Neural Network Model based on the model name to get the Image Embeddings

    params : 
        model_name      :   string - model short name
        train           :   Boolean to set the model in eval mode or not

    returns :
        Returns the PyTorch Neural Network Module for the given model architecture
    """
    if model_name == 'vgg' : 
        model = models.vgg19(pretrained = True).to(DEVICE)
    elif model_name == 'resnet50' : 
        model = models.resnet50(pretrained = True).to(DEVICE)
    elif model_name == 'resnet152' : 
        model = models.resnet152(pretrained = True).to(DEVICE)
    else : 
        raise Exception(f"The requested Model {model_name} is not Implemented yet.")
    if not train : 
        model.eval()
    return model


class SaveFeatures() : 
    """
    To get the Image Embeddings from the model using PyTorch Hooks

    """
    features=None
    def __init__(self, m : nn.Module): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module : nn.Module, input : torch.Tensor, output:torch.Tensor) : 
        """
        To process the output of the layer to which the hook is registered
        """ 
        out = output.detach().squeeze().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        """
        To remove the Hook
        """
        self.hook.remove()



def get_features(dataloader : DataLoader, sf : Callable, model : nn.Module, save:bool = False, save_path:str = None) -> np.ndarray : 
    """
    To Process the dataloader and extract the embedding features for all the Images

    params : 
        dataloader      :   DataLoader Object - contains the image data for which 
                            we need to get the embeddings
        sf              :   Class - The class that contains the hook and saved features
        model           :   Neural Network Module - model that we use for extracting embedding features
        save            :   Boolean - whether to save the features as a pickle file or not
        save_path       :   String - Pickle file path

    returns : 
        Returns the Embedding Features as a numpy array
    """
    paths = []
    with torch.no_grad() : 
        for batch in tqdm(dataloader) : 
           model(batch['image'].to(DEVICE))
           paths.extend(batch['image_path'])
    
    features = sf.features
    sf.remove()
    return paths, features

def build_annoy_tree(
    features:Union[np.ndarray, pd.Series], feature_len:int = 4096, 
    ntree:int = 500, build:bool = True, save_path:str = None
) : 
    """
    Function to Define and build the Anooy Search Tree

    params : 
        features        :   Numpy Array - contains the features to build the Annoy Tree
        feature_len     :   Int - Contains the Feature Length to build the Annoy Tree
        ntree           :   int - Number of trees to build
    """
    annoy_tree = AnnoyIndex(feature_len, metric='euclidean')
    if build : 
        for i, vector in tqdm(enumerate(features)):
            annoy_tree.add_item(i, vector)
        _  = annoy_tree.build(ntree)
    if save_path is not None : 
        annoy_tree.load(save_path)
    
    return annoy_tree



def get_similar_images_annoy(
    features_df:pd.DataFrame, img_index:int, 
    tree:Callable = None, neighbours = 50, 
    print_time = True
) -> Tuple[str, pd.DataFrame]:
    """
    To get Similar Images from Anooy Tree Based on Index

    params : 
        features_df     :   DataFrame - Contains the features and Image paths
        img_index       :   Int - Index of the image to get neighbours
        tree            :   Annoy Tree - to get the neighours
        neighbours      :   Int - Number of neoghbours to retrieve
        print_time      :   Bool - Whether to print the time taken or not

    returns : 
        Returns the tuple of Base Image path, DataFrame containing the Neighbours details 
    """

    start = time.time()
    base_img_id, _, _  = features_df.iloc[img_index]
    similar_img_ids = tree.get_nns_by_item(img_index, neighbours)
    end = time.time()
    if print_time : print(f'Took {(end - start) * 1000} ms for Neighbour Retrieval')
    return base_img_id, features_df.iloc[similar_img_ids].reset_index(drop = True)


def get_similar_images_annoy_by_feat(
    features_df:pd.DataFrame, img_vector:np.ndarray, 
    tree:Callable, neighbours = 50, print_time = True
) -> pd.DataFrame:
    """
    To get Similar Images from Anooy Tree Based on Index

    params : 
        features_df     :   DataFrame - Contains the features and Image paths
        img_vector      :   Numpy Array - Features of the image to get neighbours
        tree            :   Annoy Tree - to get the neighours
        neighbours      :   Int - Number of neoghbours to retrieve
        print_time      :   Bool - Whether to print the time taken or not

    returns : 
        Returns the DataFrame containing the Neighbours details 
    """
    
    start = time.time()
    similar_img_ids = tree.get_nns_by_vector(img_vector, neighbours)
    end = time.time()
    if print_time : print(f'Took {(end - start) * 1000} ms for Neighbour Retrieval')
    return features_df.iloc[similar_img_ids].reset_index(drop = True)


def save_image_grid(base_image:str, image_df:pd.DataFrame, path:str) -> torch.Tensor : 
    """
    To Save the Images as a Grid using PyTorch

    params : 
        image_df    :   DataFrame - Contains the Images Details
        path        :   String - Path to Save the Grid Image 
    """
    images = []
    to_tensor = transforms.ToTensor()
    if base_image is not None : 
        img = read_image(base_image, SIZE)
        img = to_tensor(img)
        images.append(img)

    for i, row in image_df.iterrows() : 
        img = read_image(os.path.join('dataset',row['ImageName']), SIZE)
        img = to_tensor(img)
        images.append(img)
    images = torch.stack(images)
    torchvision.utils.save_image(images, nrow = 10, fp = path)
    logging.info(f"Saved Neighbours Grid at {path}")
    return images


def plot_images(base_image:str = None, image_df:pd.DataFrame = None) -> str : 
    """
    To Plot The Image and its Neighbours

    params : 
        base_image      :   String - Contains the Path of Image
        image_df        :   DataFrame - Contains the details of the Image Neighbours 
    """

    if base_image is not None : 
        img = read_image(base_image, SIZE)
        plt.imshow(img)
    n_items = len(image_df) - 1
    columns = 10
    rows = int(np.ceil(n_items+1/columns))
    fig=plt.figure(figsize=(rows, 10*rows))
    for i in range(rows):
        for j in range(columns):
            try : 
                img = read_image(os.path.join('dataset', image_df.loc[i+j+1, 'ImageName']), SIZE)
                fig.add_subplot(rows, columns, i+j+1)
                plt.imshow(img)
            except : 
                pass
    plt.savefig("neighbor_plot.png")
    return  "neighbor_plot.png"

def get_similar_images(
    img_path:str, features:pd.DataFrame, sf:Callable, 
    tree:Callable, model:nn.Module, neighbours:int, 
    plot:int = 0
) -> pd.DataFrame: 
    """
    To Get Neighboursfor given Image based on its Path

    params : 
        img_path        :   String - Contains Path f Image to get Neighbours
        features        :   DataFrame - Contains the features and Image paths
        tree            :   Annoy Tree - to get the neighours
        sf              :   Callable - The Class which contains the registered PyTorch Hook and Features
        model           :   Neural Network Model - to get the Features
        neighbours      :   Int - Number of neoghbours to retrieve
        plot            :   Int - Whether to plot the Neighbours or Not

    """ 
    img = read_image(img_path, SIZE)
    img = transforms.ToTensor()(img)
    img.unsqueeze_(dim = 0)
    with torch.no_grad() :
        model.eval()
        model(img.to(DEVICE))
        out = sf.features
    
    neighbours_df = get_similar_images_annoy_by_feat(features, out, tree, neighbours = neighbours, print_time = True)
    if plot == 1 : print(f"Plot Saved at {plot_images(img_path, neighbours_df)}")
    return neighbours_df

def plot_clusters(features : np.ndarray, labels : np.ndarray, model:Callable, title : str = None, plot_centroids : bool = False)  : 

    """
    To plot the clusters for the features using the given fitted model
    
    params : 
        features        :   Input features to plot the clusters
        model           :   The model to use to find the clusters
        title           :   Title of the plot
        plot_centroids  :   Whether to plot the centroids or not

    """
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')
    plt.scatter(features[:, 0], features[:, 1], c=labels,
                s=50, cmap='viridis')
    # Plot the centroids as a white X
    
    if plot_centroids : 
        try : 
            centroids = model.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        except : 
            print("Unable to get the Centroids")
            pass
    if title is not None : plt.title(title)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("cluster_plot.jpg")


def save_pickle_object(object : Any, path:str)  : 
    """
    To save the python object as a pickle file

    params : 
        object      :   Any - Contains the python object to be saved
        path        :   String - Contains the path to save the object
    """

    with open(path, 'wb') as f : 
        pkl.dump(object, f)
    

def get_pickle_object(path:str) -> Any : 
    """
    To read the object saved as a pickle file
    
    params : 
        path        :   String - Contains the path of saved Object

    returns : 
        Return the Object
    """

    with open(path, 'rb') as f : 
        data = pkl.load(f)
    return data

def get_neighbors(features_df:pd.DataFrame, tree:Callable, n_neighbors:int, path:str) -> pd.DataFrame : 
    """
    To Extract the Neighbours for all the features and Save Them

    param : 
        features_df         :   DataFrame - Containing the Features and Image Paths
        tree                :   Annoy Tree - Already built using the Features
        n_neighbors         :   Number of Neighbours to get
        path                :   String - Path to save the Neighbors

    returns :
        Dataframe containing the Neighbors 
    """
    neighbours = {}
    for i in range(len(features_df)) : 
        sample = get_similar_images_annoy(features_df, i, tree, neighbours = n_neighbors, print_time = False)
        neighbours[features_df.loc[i, 'ImageName']] = sample[1]['ImageName'].tolist()
        if i%1000 == 0 : print(f"{i} Images Processed Successfully")

    neighbours_df = pd.DataFrame(neighbours).T
    neighbours_df.columns = ['neighbour_'+str(i) for i in neighbours_df.columns]
    neighbours_df = neighbours_df.reset_index().rename(columns = {'index' : 'image'})

    neighbours_df.to_csv(path, index = False)

    return neighbours_df