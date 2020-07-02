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
#                   num_recall_samples=gin.REQUIRED,
                   nhood_sizes=gin.REQUIRED,
                   num_interventions_per_latent_dim=gin.REQUIRED
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
#  num_recall_samples = train_ground_truth_data.data_size
  num_recall_samples = 10
  dummy_input = ground_truth_data.sample_observations(1, random_state)
  dummy_mean, dummy_var = encoder_fn(dummy_input)
  
  # Samples from the normal prior
  latent_dim = repr_transform_fn(*encoder_fn(dummy_input)).shape[-1]
  latent_shape = [num_recall_samples, latent_dim]
  latent_prior_samples_np = np.random.normal(size=latent_shape)
  
  # Ground truth samples 
  gt_train_samples = ground_truth_data.sample_observations(num_recall_samples, random_state)
  gt_train_repr = repr_transform_fn(*encoder_fn(gt_train_samples))
  gt_train_repr_mean = np.mean(gt_train_repr, axis=0) # latent_shape
  gt_train_repr_std = np.std(gt_train_repr, axis=0)
  gt_train_repr_min = np.min(gt_train_repr, axis=0)
  gt_train_repr_max = np.max(gt_train_repr, axis=0)
  
  print('\n\n\n ', gt_train_repr_mean.shape, gt_train_repr_std.shape)
  
  # The predetermined set of interventions from the estimated training prior
  fixed_trained_prior_samples_np = np.random.normal(loc=gt_train_repr_mean, 
                                              scale=gt_train_repr_std, 
                                              size=latent_shape)
  print(fixed_trained_prior_samples_np.shape, fixed_trained_prior_samples_np)
  
  result_d = {'nhoods': nhood_sizes, 
              'gt_train_repr_mean': list(gt_train_repr_mean), 
              'gt_train_repr_std': list(gt_train_repr_std), 
              'gt_train_repr_min': list(gt_train_repr_min), 
              'gt_train_repr_max': list(gt_train_repr_max)}
  sess = tf.Session()
  with sess.as_default():
    n_comp = min(num_recall_samples, 1000)

    print('\n\n\n Computing the total recall...')
    # Sample ground truth data and vae process it
    decoded_gt_samples = decoder_fn(repr_transform_fn(*encoder_fn(gt_train_samples)))
    decoded_gt_samples = decoded_gt_samples.reshape(num_recall_samples, -1)
    decoded_gt_pca = PCA(n_components=n_comp)
    reduced_decoded_gt_samples = decoded_gt_pca.fit_transform(decoded_gt_samples)

    # Generated samples from the normal prior, processed with gt PCA
    generated_prior_samples = decoder_fn(latent_prior_samples_np)
    generated_prior_samples = generated_prior_samples.reshape(num_recall_samples, -1)
    reduced_generated_prior_samples = decoded_gt_pca.transform(generated_prior_samples)
    assert(reduced_generated_prior_samples.shape == reduced_decoded_gt_samples.shape)
    
    # Generated samples from the estimated training prior, processed with gt PCA
    generated_trained_prior_samples = decoder_fn(fixed_trained_prior_samples_np)
    generated_trained_prior_samples = generated_trained_prior_samples.reshape(num_recall_samples, -1)
    reduced_generated_trained_prior_samples = decoded_gt_pca.transform(generated_trained_prior_samples)
    assert(reduced_generated_prior_samples.shape == reduced_generated_trained_prior_samples.shape)
    
    # --- Original sharp images - discarded to now 
#    gt_train_samples = gt_samples.reshape(num_recall_samples, -1)
#    gt_pca = PCA(n_components=n_comp)
#    reduced_gt_samples = gt_pca.fit_transform(gt_samples)
#    assert(reduced_gt_samples.shape == reduced_decoded_gt_samples.shape)

#    # compute model recall: gt vs generated
#    gt_generated_result = iprd.knn_precision_recall_features(
#        reduced_gt_samples, 
#        reduced_generated_prior_samples, 
#        nhood_sizes=nhood_sizes,
#        row_batch_size=500, col_batch_size=100, num_gpus=1)
#    update_result_dict(result_d, ['gt_generated_', gt_generated_result])
    # ----
    
    # compute model recall: model(gt) vs normal prior generated
    decoded_gt_prior_generated_result = iprd.knn_precision_recall_features(
        reduced_decoded_gt_samples, 
        reduced_generated_prior_samples, 
        nhood_sizes=nhood_sizes,
        row_batch_size=500, col_batch_size=100, num_gpus=1)
    update_result_dict(result_d, ['decoded_gt_prior_generated_', 
                                  decoded_gt_prior_generated_result])
    
    # compute model recall: model(gt) vs estimated training prior generated
    decoded_gt_trained_prior_generated_result = iprd.knn_precision_recall_features(
        reduced_decoded_gt_samples, 
        reduced_generated_trained_prior_samples, 
        nhood_sizes=nhood_sizes,
        row_batch_size=500, col_batch_size=100, num_gpus=1)
    update_result_dict(result_d, ['decoded_gt_trained_prior_generated_', 
                                  decoded_gt_trained_prior_generated_result])
  
    # compute model recall:normal prior generated vs estimated training prior generated
    prior_generated_trained_prior_generated_result = iprd.knn_precision_recall_features(
        reduced_generated_prior_samples, 
        reduced_generated_trained_prior_samples, 
        nhood_sizes=nhood_sizes,
        row_batch_size=500, col_batch_size=100, num_gpus=1)
    update_result_dict(result_d, ['prior_generated_trained_prior_generated_', 
                                  prior_generated_trained_prior_generated_result])
    
    # Choose a subset of interventions
    subset_interventions = np.random.choice(
          np.arange(num_recall_samples), size=num_interventions_per_latent_dim, 
          replace=False)
    result_d['subset_interventions'] = list(subset_interventions)

    # Pick a latent dimension
    for dim in range(latent_dim):
      print('\n\n\n Computing the recall for latent dim ', dim)
      
      agg_fix_one_vs_prior_generated_result = {'precision': [], 'recall': []}
      agg_fix_one_vs_trained_prior_generated_result = {'precision': [], 'recall': []}
      agg_fix_one_vs_decoded_gt_result = {'precision': [], 'recall': []}
      
      agg_vary_one_vs_prior_generated_result = {'precision': [], 'recall': []}
      agg_vary_one_vs_trained_prior_generated_result = {'precision': [], 'recall': []}
      agg_vary_one_vs_decoded_gt_result =  {'precision': [], 'recall': []}
      
      # intervene several times
      for intervention in range(num_interventions_per_latent_dim):
        inter_id = subset_interventions[intervention]
        print('n\n\n Intervention num', intervention)
        print(' Intervention row', inter_id)
        
        # --- fix one, vary the rest
        latent_intervention = float(fixed_trained_prior_samples_np[inter_id, dim])
        fix_one_latent_from_trained_prior_samples = np.copy(fixed_trained_prior_samples_np)
        fix_one_latent_from_trained_prior_samples[:, dim] = latent_intervention
        # decode the samples and trasform them with the PCA
        gen_fix_one_latent_from_trained_prior_samples = decoder_fn(
            fix_one_latent_from_trained_prior_samples).reshape(num_recall_samples, -1)
        reduced_gen_fix_one_latent_from_trained_prior_samples = decoded_gt_pca.transform(
            gen_fix_one_latent_from_trained_prior_samples)
        assert(reduced_gen_fix_one_latent_from_trained_prior_samples.shape == \
               reduced_generated_prior_samples.shape)

        
        # - Calculate the relative recall
        # compare to the generated normal prior samples
        fix_one_vs_prior_generated_result = iprd.knn_precision_recall_features(
            reduced_generated_prior_samples, 
            reduced_gen_fix_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        print(agg_fix_one_vs_prior_generated_result)
        print(fix_one_vs_prior_generated_result)
        agg_fix_one_vs_prior_generated_result = agg_recall_dict(
            agg_fix_one_vs_prior_generated_result, 
            fix_one_vs_prior_generated_result,
            intervention)
        print(agg_fix_one_vs_prior_generated_result)
        
        # compare to the generated trained prior samples
        fix_one_vs_trained_prior_generated_result = iprd.knn_precision_recall_features(
            reduced_generated_trained_prior_samples, 
            reduced_gen_fix_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        agg_fix_one_vs_trained_prior_generated_result = agg_recall_dict(
            agg_fix_one_vs_trained_prior_generated_result, 
            fix_one_vs_trained_prior_generated_result,
            intervention)
        
        # compare to the decoded gt 
        fix_one_vs_decoded_gt_result = iprd.knn_precision_recall_features(
            reduced_decoded_gt_samples, 
            reduced_gen_fix_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        agg_fix_one_vs_decoded_gt_result = agg_recall_dict(
            agg_fix_one_vs_decoded_gt_result, 
            fix_one_vs_decoded_gt_result,
            intervention)
        
        
        # --- vary one, fix the rest
        latent_variation = np.copy(fixed_trained_prior_samples_np[:, dim])
        print(latent_variation.shape, latent_variation)
        vary_one_latent_from_trained_prior_samples = np.copy(
            fixed_trained_prior_samples_np[inter_id]).reshape(1, latent_dim)
        vary_one_latent_from_trained_prior_samples = np.full(
            latent_shape, vary_one_latent_from_trained_prior_samples)
        print(vary_one_latent_from_trained_prior_samples.shape)
        print(vary_one_latent_from_trained_prior_samples)
        
        vary_one_latent_from_trained_prior_samples[:, dim] = latent_variation
        print(vary_one_latent_from_trained_prior_samples.shape)
        print(vary_one_latent_from_trained_prior_samples)
        
        # decode the samples and trasform them with the PCA
        gen_vary_one_latent_from_trained_prior_samples = decoder_fn(
            vary_one_latent_from_trained_prior_samples).reshape(num_recall_samples, -1)
        reduced_gen_vary_one_latent_from_trained_prior_samples = decoded_gt_pca.transform(
            gen_vary_one_latent_from_trained_prior_samples)
        assert(reduced_gen_vary_one_latent_from_trained_prior_samples.shape == \
               reduced_generated_prior_samples.shape)
        
        # - Calculate the recall
        # compare to the generated normal prior samples
        vary_one_vs_prior_generated_result = iprd.knn_precision_recall_features(
            reduced_generated_prior_samples, 
            reduced_gen_vary_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        print(agg_vary_one_vs_prior_generated_result)
        print(vary_one_vs_prior_generated_result)
        agg_vary_one_vs_prior_generated_result = agg_recall_dict(
            agg_vary_one_vs_prior_generated_result, 
            vary_one_vs_prior_generated_result,
            intervention)
        print(agg_vary_one_vs_prior_generated_result)
        
        # compare to the generated trained prior samples
        vary_one_vs_trained_prior_generated_result = iprd.knn_precision_recall_features(
            reduced_generated_trained_prior_samples, 
            reduced_gen_vary_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        agg_vary_one_vs_trained_prior_generated_result = agg_recall_dict(
            agg_vary_one_vs_trained_prior_generated_result, 
            vary_one_vs_trained_prior_generated_result,
            intervention)
        
        # compare to the decoded gt 
        vary_one_vs_decoded_gt_result = iprd.knn_precision_recall_features(
            reduced_decoded_gt_samples, 
            reduced_gen_vary_one_latent_from_trained_prior_samples,
            nhood_sizes=nhood_sizes,
            row_batch_size=500, col_batch_size=100, num_gpus=1)
        agg_vary_one_vs_decoded_gt_result = agg_recall_dict(
            agg_vary_one_vs_decoded_gt_result, 
            vary_one_vs_decoded_gt_result,
            intervention)
        
      update_result_dict_with_agg(
          result_d, 
          [str(dim) + '_fix_one_vs_prior_generated_', agg_fix_one_vs_prior_generated_result],
          [str(dim) + '_fix_one_vs_trained_prior_generated_', agg_fix_one_vs_trained_prior_generated_result],
          [str(dim) + '_fix_one_vs_decoded_gt_', agg_fix_one_vs_decoded_gt_result],
          [str(dim) + '_vary_one_vs_prior_generated_', agg_vary_one_vs_prior_generated_result], 
          [str(dim) + '_vary_one_vs_trained_prior_generated_', agg_vary_one_vs_trained_prior_generated_result],
          [str(dim) + '_vary_one_vs_decoded_gt_', agg_vary_one_vs_decoded_gt_result])
  print(result_d)
  return result_d
    

def update_result_dict(result_d, *args):
  for arg in args:
    update_key = arg[0]
    update_d = {update_key + key: list(value) for key, value in arg[1].items()}
    result_d.update(update_d)
  return result_d

def update_result_dict_with_agg(result_d, *args):
  for arg in args:
    print(arg)
    update_key = arg[0]
    update_d = {update_key + key: list(np.mean(value, axis=0)) for key, value in arg[1].items()}
    if 'fix_one' in update_key:
      update_d = {update_key + 'recall_sum': list(np.sum(arg[1]['recall'], axis=0))}
    result_d.update(update_d)
  return result_d

def agg_recall_dict(agg_d, new_d, inter_id):
  if inter_id == 0:
    return new_d
  else:
    for k, v in new_d.items():
      agg_d[k] = np.vstack([agg_d[k], new_d[k]])
    return agg_d
 