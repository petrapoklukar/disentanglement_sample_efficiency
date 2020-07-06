#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:58:31 2020

@author: petrapoklukar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin.tf
import os
from disentanglement_lib.data.ground_truth import named_data
import numpy as np
import pickle
from sklearn.decomposition import PCA

def train_pca_with_gin(model_dir,
                       overwrite=False,
                       gin_config_files=None,
                       gin_bindings=None):
  """Trains a PCA based on the provided gin configuration.

  This function will set the provided gin bindings, call the train() function
  and clear the gin config. Please see train() for required gin bindings.

  Args:
    model_dir: String with path to directory where model output should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  train_pca(model_dir, overwrite)
  gin.clear_config()

  
@gin.configurable("train_pca", blacklist=["model_dir", "overwrite"])
def train_pca(model_dir,
              overwrite=False,
              random_seed=gin.REQUIRED,
              num_pca_components=gin.REQUIRED,
              name="",
              model_num=None):
  """Trains the pca and saves it.

  Args:
    model_dir: String with path to directory where model output should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    random_seed: Integer with random seed used for training.
    num_pca_components: list with the number of pca components.
    name: Optional string with name of the model (can be used to name models).
    model_num: Optional integer with model number (can be used to identify
      models).
  """
  # Obtain the datasets.
  dataset_train, _ = named_data.get_named_ground_truth_data()

  # Create a numpy random state. We will sample the random seeds for training
  # and evaluation from this.
  random_state = np.random.RandomState(random_seed)
  
  # Save the pca
  pca_dir = os.path.join(model_dir, "pca")
  for num_comp in num_pca_components:
    pca = PCA(n_components=num_comp, random_state=random_state)
    original_train_images = dataset_train.images.reshape(dataset_train.data_size, -1)
    print(original_train_images.shape)
    trained_pca = pca.fit(original_train_images)

    pca_model_name = 'pca_{0}_{1}comp.pkl'.format(dataset_train.name, str(num_comp))
    pca_export_path = os.path.join(pca_dir, pca_model_name)
    with open(pca_export_path, 'wb') as f:
      pickle.dump(trained_pca, f)
    
    
    
    
    
    
  