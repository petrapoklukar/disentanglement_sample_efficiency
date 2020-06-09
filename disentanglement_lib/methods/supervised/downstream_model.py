#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:40:18 2020

@author: petrapoklukar

Defines a supervised decoder used in downstream tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import gin.tf


class DownstreamModel(object):
  """Abstract base class of a Gaussian encoder model."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function used for training/evaluation."""
    raise NotImplementedError()

  def forward_pass(self, input_tensor, observation_shape, is_training):
    """Applies the Gaussian encoder to images.

    Args:
      input_tensor: Input Tensor to be processed.
      is_training: Boolean indicating whether in training mode.

    Returns:
      Tuple of tensors with the mean and log variance of the Gaussian encoder.
    """
    raise NotImplementedError()


@gin.configurable("export_as_tf_hub", whitelist=[])
def export_as_tf_hub(downstream_model,
                     representation_shape,
                     observation_shape,
                     checkpoint_path,
                     export_path,
                     drop_collections=None):
  """Exports the provided GaussianEncoderModel as a TFHub module.

  Args:
    gaussian_encoder_model: GaussianEncoderModel to be exported.
    observation_shape: Tuple with the observations shape.
    checkpoint_path: String with path where to load weights from.
    export_path: String with path where to save the TFHub module to.
    drop_collections: List of collections to drop from the graph.
  """

  def module_fn(is_training):
    """Module function used for TFHub export."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      # Add a signature for the Gaussian encoder.
      representation_placeholder = tf.placeholder(
          dtype=tf.float32, shape=[None] + representation_shape)
      reconstructed_images = downstream_model.forward_pass(
          representation_placeholder, observation_shape, is_training)
      hub.add_signature(
          name="reconstructions",
          inputs={"representations": representation_placeholder},
          outputs={"images": reconstructed_images})

  # Export the module.
  # Two versions of the model are exported:
  #   - one for "test" mode (the default tag)
  #   - one for "training" mode ("is_training" tag)
  # In the case that the encoder/decoder have dropout, or BN layers, these two
  # graphs are different.
  tags_and_args = [
      ({"train"}, {"is_training": True}),
      (set(), {"is_training": False}),
  ]
  spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args,
                                drop_collections=drop_collections)
  spec.export(export_path, checkpoint_path=checkpoint_path)
