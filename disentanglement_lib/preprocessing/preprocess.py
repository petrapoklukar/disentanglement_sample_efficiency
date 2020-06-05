#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:27:47 2020

@author: petrapoklukar

Preprocessing step that creates train and validation splits.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import gin.tf


def preprocess_with_gin(dataset_name,
                        overwrite=False,
                        gin_config_files=None,
                        gin_bindings=None):
  """Postprocess a trained model based on the provided gin configuration.

  This function will set the provided gin bindings, call the postprocess()
  function and clear the gin config. Please see the postprocess() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the model is saved.
    output_dir: String with the path where the representation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  preprocess(dataset_name, overwrite)
  gin.clear_config()


@gin.configurable(
    "preprocess", blacklist=["dataset_name", "overwrite"])
def preprocess(dataset_name,
               overwrite=False,
               preprocess_fn=gin.REQUIRED,
               random_seed=gin.REQUIRED,
               name=""):
  """Loads a trained Gaussian encoder and extracts representation.

  Args:
    dataset_name: String with dataset name to split into train and validation.
    overwrite: Boolean indicating whether to overwrite output directory.
    preprocess_fn: Function used to split the dataset.
    random_seed: Integer with random seed used for postprocessing (may be
      unused).
    name: Optional string with name of the representation (can be used to name
      representations).
  """
  # We do not use the variable 'name'. Instead, it can be used to name
  # representations as it will be part of the saved gin config.
  del name
  preprocess_fn(dataset_name, np.random.RandomState(random_seed))
  