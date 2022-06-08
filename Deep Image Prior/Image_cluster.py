# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:57:20 2021

@author: sg
"""

import cv2
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import glob
import re
from collections import defaultdict
import sys
import random 

 	
import os
path = os.getcwd()

def get_image_fnames(path):
    """
    Return ist of needed images
    """
    fnames = list(glob.glob(f"{img_dir}*.png"))
    return fnames


def combine_images_into_tensor(img_fnames, size=160):
    """
    Given a list of image filenames, read the images, flatten them
    and return a tensor such that each row contains one image.
    Size of individual image: 320*320
    """
    # Initialize the tensor
    tensor = np.zeros((len(img_fnames), size * size))

    for i, fname in enumerate(img_fnames):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        tensor[i] = img.reshape(size * size)

    return tensor


def get_pca_reducer_incremental(tr_tensor, n_comp=5):
    # Apply Incremental PCA on the training images
    pca = IncrementalPCA(n_components=n_comp, batch_size=25)

    for i in range(0, len(tr_tensor), 25):
        print(f"fitting {i//25} th batch")
        pca.partial_fit(tr_tensor[i:i+25, :])

    return pca


def cluster_images(all_img_fnames, num_clusters=4):
    # Select images at random for PCA
    random.shuffle(all_img_fnames)
    tr_img_fnames = all_img_fnames[:400]

    # Flatten and combine the images
    tr_tensor = combine_images_into_tensor(tr_img_fnames)

    # Perform PCA
    print("Learning PCA...")
    n_comp = 5
    pca = get_pca_reducer_incremental(tr_tensor, n_comp)

    # Transform images in batches
    print("applying PCA transformation")
    points = np.zeros((len(all_img_fnames), n_comp))
    batch_size = 50
    for i in range(0, len(all_img_fnames), batch_size):
        print(f"Transforming {i//25} th batch")
        batch_fnames = all_img_fnames[i:i+batch_size]
        all_tensor = combine_images_into_tensor(batch_fnames)
        points[i:i+batch_size] = pca.transform(all_tensor)

    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)

    # Organize image filenames based on the obtained clusters
    cluster_fnames = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        cluster_fnames[label].append(all_img_fnames[i])

    return cluster_fnames


if __name__ == "__main__":
    # Directory containing images that need to be clustered
    img_dir = ''

    # Balance the images
    all_img_fnames = get_image_fnames(img_dir)
    clustered_fnames = cluster_images(all_img_fnames, num_clusters=2)