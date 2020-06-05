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

@gin.configurable("split_train_and_validation", 
                  blacklist=["dataset_name", "random_state"])
def create_split_train_and_validation(dataset_name, 
                                      random_state, 
                                      unit_labels=False):
    """ Randomly splits the model split into smaller datasets of different
        sizes.
    
    Args:   
        filename: name of the file to split further
    """
    SHAPES3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", 
            dataset_name + ".h5")
    dataset_split = h5py.File(SHAPES3D_PATH, 'r')
    print(dataset_split.keys())
    images_split = dataset_split['images'][()]
    labels_split = dataset_split['labels'][()]
    indices_split = dataset_split['indices'][()]
    dataset_size = len(images_split)
    
    ims = np.array(images_split)
    labs = np.array(labels_split)
    inds = np.array(indices_split)
    
    if unit_labels:
        labels_min = np.array([0., 0., 0., 0.75, 0., -30.])
        labels_max = np.array([0.9, 0.9, 0.9, 1.25, 3., 30.])
        labels_split = (labels_split - labels_min)/(labels_max - labels_min)
        print(labels_split.shape)
        assert(np.min(labels_split) == 0 and np.max(labels_split) == 1)
    
    all_local_indices = random_state.choice(dataset_size, dataset_size, replace=False)
    random_state.shuffle(all_local_indices)
    splitratio = int(dataset_size * 0.85)

    train_local_indices = all_local_indices[:splitratio]
    test_local_indices = all_local_indices[splitratio:]
    
    print('Writing files')
    for indices, split in list(zip([train_local_indices, test_local_indices], 
                                      ['_train', '_valid'])):
        
        SPLIT_SHAPES3D_PATH = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", 
            dataset_name + split + ".h5")
        assert(ims[indices].shape[0] == indices.shape[0])
        assert(labs[indices].shape[0] == indices.shape[0])
        assert(inds[indices].shape[0] == indices.shape[0])
        hf = h5py.File(SPLIT_SHAPES3D_PATH, 'w')
        hf.create_dataset('images', data=ims[indices])
        hf.create_dataset('labels', data=labs[indices])
        hf.create_dataset('indices', data=inds[indices])
        hf.close()
        
    dataset_split.close()

    

    

