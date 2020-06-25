#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:50:24 2020

@author: petrapoklukar

Computes recall metric.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation.metrics import iprd_score as iprd 
import numpy as np
from six.moves import range
import gin.tf
from disentanglement_lib.data.ground_truth import named_data
import tensorflow as tf
from sklearn.decomposition import PCA


@gin.configurable(
    "recall",
    blacklist=["ground_truth_data", "encoder_fn", "repr_transform_fn",
               "decoder_fn", "random_state", "artifact_dir"])
def compute_recall(ground_truth_data,
                   encoder_fn, 
                   repr_transform_fn,
                   decoder_fn,
                   random_state,
                   artifact_dir=None,
                   num_recall_samples=gin.REQUIRED,
                   nhood_sizes=gin.REQUIRED
                   ):
  """TBA

  Args:
    ground_truth_data: GroundTruthData to be sampled from.


    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
  """
  del artifact_dir
  train_ground_truth_data, test_ground_truth_data = ground_truth_data
  ground_truth_data = train_ground_truth_data
  dummy_input = ground_truth_data.sample_observations(1, random_state)
  dummy_mean, dummy_var = encoder_fn(dummy_input)
  # Samples from the prior
  print(dummy_mean.shape)
  dummy_repr = repr_transform_fn(dummy_mean, dummy_var)
  print(dummy_repr.shape)
  latent_shape = [num_recall_samples, int(dummy_repr.shape[-1])]
  print(latent_shape)
  latent_prior_samples = tf.random_normal(latent_shape, 0, 1, name="decoder/latent_vectors")
  print(latent_prior_samples.shape)
  
  sess = tf.Session()
  with sess.as_default():
    # Generated samples
    generated_prior_samples = decoder_fn(latent_prior_samples)
    print(generated_prior_samples.shape)
    generated_prior_samples = tf.reshape(generated_prior_samples, [num_recall_samples, -1]).eval()
    generated_pca = PCA(n_components=1000)
    reduced_generated_prior_samples = generated_pca.fit_transform(generated_prior_samples)
    
    # Sample ground truth data
    gt_samples = ground_truth_data.sample_observations(num_recall_samples, random_state)
    print(gt_samples.shape)
    decoded_gt_samples = decoder_fn(encoder_fn(gt_samples))
    print(decoded_gt_samples.shape)
    
    decoded_gt_samples = tf.reshape(decoded_gt_samples, [num_recall_samples, -1]).eval()
    decoded_gt_pca = PCA(n_components=1000)
    reduced_decoded_gt_samples = decoded_gt_pca.fit_transform(decoded_gt_samples)

    gt_samples = tf.reshape(gt_samples, [num_recall_samples, -1]).eval()
    print(gt_samples.shape)  
    gt_pca = PCA(n_components=1000)
    reduced_gt_samples = gt_pca.fit_transform(gt_samples)

    # compute model recall: gt vs generated
    gt_generated_result = iprd.knn_precision_recall_features(
                                  reduced_gt_samples, 
                                  reduced_generated_prior_samples, 
                                  nhood_sizes=nhood_sizes,
                                  row_batch_size=500, col_batch_size=100, 
                                  num_gpus=1)
    
    # compute model recall: model(gt) vs generated
    decoded_gt_generated_result = iprd.knn_precision_recall_features(
                                  reduced_decoded_gt_samples, 
                                  reduced_generated_prior_samples, 
                                  nhood_sizes=nhood_sizes,
                                  row_batch_size=500, col_batch_size=100, 
                                  num_gpus=1)
  result_d = {
      'gt_generated_result': gt_generated_result,
      'decoded_gt_generated_result': decoded_gt_generated_result
      }
  return result_d
    
  
  
 