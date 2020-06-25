#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:16:50 2020

@author: petrapoklukar

Calculating the recall.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import os
import time
import warnings
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.postprocessing import methods  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import recall
from disentanglement_lib.utils import convolute_hub
from disentanglement_lib.utils import results
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import gin.tf


def evaluate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
  """Calculate precision and recall of a trained model based on the provided 
  gin configuration.

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
  evaluate(model_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "evaluate_with_decodings", blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate(model_dir,
             output_dir,
             overwrite=False,
             postprocess_fn=gin.REQUIRED,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):
  """Loads a trained Gaussian encoder and decoder.

  Args:
    model_dir: String with path to directory where the model is saved.
    output_dir: String with the path where the representation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    postprocess_fn: Function used to extract the representation (see methods.py
      for examples).
    random_seed: Integer with random seed used for postprocessing (may be
      unused).
    name: Optional string with name of the representation (can be used to name
      representations).
  """
  # We do not use the variable 'name'. Instead, it can be used to name
  # representations as it will be part of the saved gin config.
  del name

  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Set up timer to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dir, "results", "gin", "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
          "'", ""))
  dataset = named_data.get_named_ground_truth_data()
  print(dataset[0].name, dataset[1].name)
  
  # Path to TFHub module of previously trained model.
  module_path = os.path.join(model_dir, "tfhub")
  with hub.eval_function_for_module(module_path) as f:

    def _gaussian_encoder(x):
      """Encodes images using trained model."""
      # Push images through the TFHub module.
      output = f(dict(images=x), signature="gaussian_encoder", as_dict=True)
      # Convert to numpy arrays and return.
      return {key: np.array(values) for key, values in output.items()}
    
    def _decoder(z):
      """Encodes images using trained model."""
      # Push images through the TFHub module.
      output = f(dict(latent_vectors=z), signature="decoder", as_dict=True)
      # Convert to numpy arrays and return.
      return np.array(output['images'])

    # Run the postprocessing function which returns a transformation function
    # that can be used to create the representation from the mean and log
    # variance of the Gaussian distribution given by the encoder. Also returns
    # path to a checkpoint if the transformation requires variables.
    transform_fn, transform_checkpoint_path = postprocess_fn(
        dataset, _gaussian_encoder, np.random.RandomState(random_seed),
        output_dir)

    print('\n\n\n Calculating recall')
    # Computes scores of the representation based on the evaluation_fn.
    if _has_kwarg_or_kwargs(evaluation_fn, "artifact_dir"):
      artifact_dir = os.path.join(model_dir, "artifacts")
      results_dict = evaluation_fn(
          dataset,
          transform_fn,
          _decoder,
          random_state=np.random.RandomState(random_seed),
          artifact_dir=artifact_dir)
    else:
      # Legacy code path to allow for old evaluation metrics.
      warnings.warn(
          "Evaluation function does not appear to accept an"
          " `artifact_dir` argument. This may not be compatible with "
          "future versions.", DeprecationWarning)
      results_dict = evaluation_fn(
          dataset,
          transform_fn,
          _decoder,
          random_state=np.random.RandomState(random_seed))

  # Save the results (and all previous results in the pipeline) on disk.
  original_results_dir = os.path.join(model_dir, "results")
  results_dir = os.path.join(output_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "evaluation", results_dict,
                                  original_results_dir)


def _has_kwarg_or_kwargs(f, kwarg):
  """Checks if the function has the provided kwarg or **kwargs."""
  # For gin wrapped functions, we need to consider the wrapped function.
  if hasattr(f, "__wrapped__"):
    f = f.__wrapped__
  args, _, kwargs, _ = inspect.getargspec(f)
  if kwarg in args or kwargs is not None:
    return True
  return False
