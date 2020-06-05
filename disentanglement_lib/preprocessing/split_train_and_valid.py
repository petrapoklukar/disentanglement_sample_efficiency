#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:41:20 2020

@author: petrapoklukar
"""

import numpy as np
import h5py
import os
import gin.tf

@gin.configurable("split_train_and_validation")
def create_split_train_and_validation(dataset_name, random_state):
    """ Randomly splits the model split into smaller datasets of different
        sizes.
    
    Args:   
        filename: name of the file to split further
    """
    SHAPES3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", dataset_name + ".h5")
    dataset_split = h5py.File(SHAPES3D_PATH, 'r')
    print(dataset_split.keys())
    images_split = dataset_split['images'][()]
    labels_split = dataset_split['labels'][()]
    indices_split = dataset_split['indices'][()]
    dataset_size = len(images_split)
    
    print(dataset_size)
    all_local_indices = np.arange(dataset_size, random_state=random_state)
    all_local_indices = random_state.shuffle(all_local_indices)
    splitratio = int(len(dataset_size) * 0.15)

    train_local_indices = all_local_indices[:splitratio]
    test_local_indices = all_local_indices[splitratio:]
    
    images_split_train = images_split[train_local_indices]
    labels_split_train = labels_split[train_local_indices]
    indices_split_train = indices_split[train_local_indices]
    
    images_split_test = images_split[test_local_indices]
    labels_split_test = labels_split[test_local_indices]
    indices_split_test = indices_split[test_local_indices]
    
    split_indices = []
    for split_size in split_sizes:
        model_split_indices = all_indices[:split_size]    
        assert(model_split_indices.shape[0] == split_size)
        split_indices.append(model_split_indices)
    
    assert(np.intersect1d(split_indices[0], 
                          split_indices[num_splits]).size == split_sizes[0] \
            for num_splits in range(len(split_sizes)))
    
    savename_list = []
    for split_size in range(len(split_sizes)): 
        savename_list.append('3dshapes_model_s{0}'.format(split_sizes[split_size]))
        
    write_datasets(split_indices, savename_list, images_split, labels_split, indices_split)
    dataset_split.close()
