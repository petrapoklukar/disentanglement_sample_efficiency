#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:43:48 2020

@author: petrapoklukar

Main training protocol used for supervised disentanglement models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.methods.supervised import downstream_model
from disentanglement_lib.methods.supervised import decoder  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf


def train_with_gin(model_dir,
                   trained_model_dir, 
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None):
  """Trains a model based on the provided gin configuration.

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
  train(model_dir, trained_model_dir, overwrite)
  gin.clear_config()


@gin.configurable("model", blacklist=["model_dir", "overwrite"])
def train(model_dir,
          trained_model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          random_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          eval_steps=1000,
          name="",
          model_num=None):
  """Trains the estimator and exports the snapshot and the gin config.

  The use of this function requires the gin binding 'dataset.name' to be
  specified as that determines the data set used for training.

  Args:
    model_dir: String with path to directory where model output should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    model: GaussianEncoderModel that should be trained and exported.
    training_steps: Integer with number of training steps.
    random_seed: Integer with random seed used for training.
    batch_size: Integer with the batch size.
    eval_steps: Optional integer with number of steps used for evaluation.
    name: Optional string with name of the model (can be used to name models).
    model_num: Optional integer with model number (can be used to identify
      models).
  """
  # We do not use the variables 'name' and 'model_num'. Instead, they can be
  # used to name results as they will be part of the saved gin config.
  del name, model_num

  # Path to TFHub module of previously trained representation.
  trained_module_path = os.path.join(trained_model_dir, "tfhub")
  with hub.eval_function_for_module(trained_module_path) as f:

    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])


  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(model_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(model_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Create a numpy random state. We will sample the random seeds for training
  # and evaluation from this.
  random_state = np.random.RandomState(random_seed)

  # Obtain the dataset.
  dataset_train, dataset_valid = named_data.get_named_ground_truth_data()

  # We create a TPUEstimator based on the provided model. This is primarily so
  # that we could switch to TPU training in the future. For now, we train
  # locally on GPUs.
  run_config = tf.contrib.tpu.RunConfig(
      tf_random_seed=random_seed,
      keep_checkpoint_max=1,
      tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=500))
  tpu_estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model.model_fn,
      model_dir=os.path.join(model_dir, "tf_checkpoint"),
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      config=run_config)

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Do the actual training.
  tpu_estimator.train(
      input_fn=_make_downstream_input_fn(dataset_train, _representation_function, 
                                         random_state.randint(2**32)),
                                         steps=training_steps)

  # Save model as a TFHub module.
  output_shape = named_data.get_named_ground_truth_data()[0].observation_shape
  module_export_path = os.path.join(model_dir, "tfhub")
  downstream_model.export_as_tf_hub(model, output_shape,
                                    tpu_estimator.latest_checkpoint(),
                                    module_export_path)

  # Save the results. The result dir will contain all the results and config
  # files that we copied along, as we progress in the pipeline. The idea is that
  # these files will be available for analysis at the end.
  results_dict = tpu_estimator.evaluate(
      input_fn=_make_downstream_input_fn(
          dataset_valid, random_state.randint(2**32), _representation_function, 
          num_batches=eval_steps))
  results_dir = os.path.join(model_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "evaluate", results_dict)


def _make_downstream_input_fn(ground_truth_data, representation_fn, seed, num_batches=None):
  """Creates an input function for the experiments."""

  def load_dataset(params):
    """TPUEstimator compatible input fuction."""
    dataset = util.tf_labeled_data_set_from_ground_truth_data(ground_truth_data, 
                                                      representation_fn, seed)
    batch_size = params["batch_size"]
    # We need to drop the remainder as otherwise we lose the batch size in the
    # tensor shape. This has no effect as our data set is infinite.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if num_batches is not None:
      dataset = dataset.take(num_batches)
    return dataset.make_one_shot_iterator().get_next()

  return load_dataset
