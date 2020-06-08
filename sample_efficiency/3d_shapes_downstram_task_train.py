#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:17:29 2020

@author: petrapoklukar
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('..')
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.supervised import train_partial
from disentanglement_lib.methods.supervised import decoder
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
import gin.tf

base_path = "3d_shape_out"
overwrite = True
path_vae = os.path.join(base_path, "vae3dshapes_model_s1000")
vae_gin_bindings = [
      "model.random_seed = 0",
      "dataset.name = '3dshapes_model_s1000'"
    ]
#train_partial.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["3d_shape_vae.gin"], 
#                             gin_bindings=vae_gin_bindings)


# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.
if False:
    postprocess_gin_bindings = [
        "postprocess.postprocess_fn = @mean_representation",
        "dataset.name='dummy_data'", 
        "postprocess.random_seed = 0"]
    for path in [path_vae]:
      representation_path = os.path.join(path, "representation")
      model_path = os.path.join(path, "model")
      # postprocess.postprocess_with_gin defines the standard extraction protocol.
      postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                       gin_config_files=None, gin_bindings=postprocess_gin_bindings)


# 4. Train a downstream task
downstream_reconstruction_train_gin_bindings = [
    "model.model = @downstream_decoder()",
    "model.batch_size = 64",
    "model.training_steps = 5", 
    "model.random_seed = 0",
    "dataset.name='3dshapes_task_s1000'",
    "decoder_optimizer.optimizer_fn = @AdamOptimizer",
    "AdamOptimizer.beta1 = 0.9", 
    "AdamOptimizer.beta2 = 0.999", 
    "AdamOptimizer.epsilon = 1e-08", 
    "AdamOptimizer.learning_rate = 0.0001", 
    "AdamOptimizer.name = 'Adam'", 
    "AdamOptimizer.use_locking = False",
    "decoder.decoder_fn = @deconv_decoder",
    "reconstruction_loss.loss_fn = @l2_loss"
]

for path in [path_vae]:
  result_path = os.path.join(path, "metrics", "TEST_reconstruction")
  representation_path = os.path.join(path, "representation")
  train_partial.train_with_gin(
      result_path, representation_path, overwrite, 
      gin_bindings=downstream_reconstruction_train_gin_bindings)#["3d_shape_classifier.gin"])#gin_bindings=gin_bindings)

pa = 1/0





downstream_train_gin_bindings = [
        "evaluation.evaluation_fn = @downstream_regression_on_representations",
        "dataset.name = '3dshapes_task'",
        "evaluation.random_seed = 111",
        "downstream_regression_on_representations.num_train = [127500]",
        "downstream_regression_on_representations.num_test = 22500",
        "predictor.predictor_fn = @mlp_regressor",
        "mlp_regressor.hidden_layer_sizes = [32, 16]",
        "mlp_regressor.activation = 'logistic'",
        "mlp_regressor.max_iter = 100",
        "mlp_regressor.random_state = 0"
    ]
    
for path in [path_vae]:
  result_path = os.path.join(path, "metrics", "TEST_factor_regression")
  representation_path = os.path.join(path, "representation")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=downstream_train_gin_bindings)#["3d_shape_classifier.gin"])#gin_bindings=gin_bindings)

pa = 1/0
