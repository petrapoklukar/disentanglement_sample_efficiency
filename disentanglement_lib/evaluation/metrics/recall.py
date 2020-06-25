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
  latent_dim = repr_transform_fn(*encoder_fn(dummy_input)).shape[-1]
  latent_shape = [num_recall_samples, latent_dim]

  result_d = {'nhoods': nhood_sizes}
  sess = tf.Session()
  with sess.as_default():
    n_comp = min(num_recall_samples, 1000)
    latent_prior_samples = tf.random_normal(
        latent_shape, 0, 1, name="decoder/latent_vectors").eval() # [num_recall_samples, 64, 64, 3]

    # Generated samples
    generated_prior_samples = decoder_fn(latent_prior_samples)
    generated_prior_samples = generated_prior_samples.reshape(num_recall_samples, -1)
    generated_pca = PCA(n_components=n_comp)
    reduced_generated_prior_samples = generated_pca.fit_transform(generated_prior_samples)
    
    # Sample ground truth data
    gt_samples = ground_truth_data.sample_observations(num_recall_samples, random_state)
    decoded_gt_samples = decoder_fn(repr_transform_fn(*encoder_fn(gt_samples)))
    
    decoded_gt_samples = decoded_gt_samples.reshape(num_recall_samples, -1)
    decoded_gt_pca = PCA(n_components=n_comp)
    reduced_decoded_gt_samples = decoded_gt_pca.fit_transform(decoded_gt_samples)

    gt_samples = gt_samples.reshape(num_recall_samples, -1)
    gt_pca = PCA(n_components=n_comp)
    reduced_gt_samples = gt_pca.fit_transform(gt_samples)

    # compute model recall: gt vs generated
    gt_generated_result = iprd.knn_precision_recall_features(
                                  reduced_gt_samples, 
                                  reduced_generated_prior_samples, 
                                  nhood_sizes=nhood_sizes,
                                  row_batch_size=500, col_batch_size=100, 
                                  num_gpus=1)
    update_result_dict(result_d, ['gt_generated_', gt_generated_result])
    
    # compute model recall: model(gt) vs generated
    decoded_gt_generated_result = iprd.knn_precision_recall_features(
                                  reduced_decoded_gt_samples, 
                                  reduced_generated_prior_samples, 
                                  nhood_sizes=nhood_sizes,
                                  row_batch_size=500, col_batch_size=100, 
                                  num_gpus=1)
    update_result_dict(result_d, ['decoded_gt_generated_', decoded_gt_generated_result])
      
  print(result_d)
  return result_d
    

def update_result_dict(result_d, *args):
  for arg in args:
    print(arg)
    update_key = arg[0]
    update_d = {update_key + key: value for key, value in arg[1].items()}
    result_d.update(update_d)
  return result_d
   
 