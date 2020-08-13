# Avantari-ML-Challenge
Retrieving Similar Images from a single Image

# Problem Statement
You are provided with a dataset of ~5k 512x512 images, your program should accept an
512x512 input image and return N images from the provided dataset similar to the input image.
## Link to the dataset
https://drive.google.com/file/d/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri/view?usp=sharing
## Evaluation Method
<p>Your code submission will be evaluated based code quality and on how accurate it is
able to find similar images</p>
<p>  simple score of C/N</p>
<p>   C = no. of correct similar images returned</p>
<p>   N = no. requested images</p>
<p>Plus points, for finding similar images with respect to unique feature</p>
<p>  simple score of F/N</p>
<p>    F = no. of images returned with the unique feature specific to the input image</p>
<p>    N = no. requested images</p>
<p>Bonus points, if the provided dataset was clustered into K groups</p>
<p>Quality of Code based on Modularity, Reusability, Maintainability, Readability</p>

# COLAB
[link to Colab!](https://colab.research.google.com/drive/1v1DTT22hSYQ1x9cwQQ3SRIEdaeRVKJTz?usp=sharing)

# Reproducing The Results

<p>To get the Features run feature.py</p>
<p>python feature.py --save_feature_path "features.pkl" --images_path "dataset/*.jpg" --model_name vgg</p>

<p>To Build the Annoy Tree and get Neighbours for all Images run tree.py</p>
<p>python tree.py --feature_path "features.pkl" --model_name vgg --tree_path "annoy_tree.ann" --neighbors_path "neighbors.csv" --n_neighbours 100</p>

<p>To Build the Clusters from Features run cluster.py</p>
<p>python cluster.py --feature_path "features.pkl" --kmeans_path "kmeans.pkl" --image_cluster_path 'image_cluster.csv' --tsne_path "tsne.pkl" --tsne_feature_path "tsne_reduced_features.pkl"
  
 <p>To get the Neighbors for New Image run neighbors.py</p>
 <p>!python neighbors.py --feature_path "features.pkl" --image_path "dataset/1008.jpg" --tree_path "annoy_tree.ann" --model_name vgg --save_neighbours 1</p>
 
# Results
## Neighbours Image 
<p>First Image Represents the Query Image and Others Represent the Neighbours</p>

![Image of Neighbours](https://octodex.github.com/images/yaktocat.png)

## Reduced Feature Plot by TSNE

## Clusters Plot by Kmeans on Reduced Features


