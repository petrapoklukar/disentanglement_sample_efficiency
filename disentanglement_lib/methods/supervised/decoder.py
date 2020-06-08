#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:57:07 2020

@author: petrapoklukar

Implementation of decoder model for supervised evaluation of disentangled
representations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.supervised import downstream_model
from six.moves import range
from six.moves import zip
import tensorflow as tf
import gin.tf


@gin.configurable("downstream_decoder")
class Decoder(downstream_model.DownstreamModel):
  """Decoder."""
  
  def __init__(self):
    super(Decoder, self).__init__()

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    output_shape = labels.get_shape().as_list()[1:]
    reconstructions = self.foward_pass(features, output_shape, is_training=is_training)
    per_sample_loss = losses.make_reconstruction_loss(labels, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = optimizers.make_vae_optimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(
          loss=reconstruction_loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)

      logging_hook = tf.train.LoggingTensorHook({
          "reconstruction_loss": reconstruction_loss,
      },
                                                every_n_iter=100)
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=reconstruction_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=reconstruction_loss,
          eval_metrics=(make_metric_fn("reconstruction_loss"), 
                        [reconstruction_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")

  def foward_pass(self, latent_tensor, observation_shape, is_training):
    """Decodes the latent_tensor to an observation."""
    return architectures.make_decoder(
        latent_tensor, observation_shape, is_training=is_training)


def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn


