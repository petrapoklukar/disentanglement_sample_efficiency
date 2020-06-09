#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import sys
sys.path.append('..')
from disentanglement_lib.methods.unsupervised import train_partial as unsupervised_train_partial
from disentanglement_lib.methods.supervised import train_partial as supervised_train_partial
from disentanglement_lib.preprocessing import preprocess
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.evaluation import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "vae model to use")
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")
flags.DEFINE_integer("rng", 0,
                     "random seed")


def main(unused_argv):

#  datasets = ["3dshapes_model_s1000", "3dshapes_model_s10000", 
#              "3dshapes_model_s50000", "3dshapes_model_s100000",
#              "3dshapes_model_s150000", "3dshapes_model_s250000"]
  base_path = "3dshapes_models"
  
  
#  for dataset in datasets:
  preproces_gin_bindings = [
        "dataset.name = '%s'" %(FLAGS.dataset),
        "preprocess.preprocess_fn = @split_train_and_validation",
        "preprocess.random_seed = %d" %(FLAGS.rng)
  ]
  preprocess.preprocess_with_gin(FLAGS.dataset,
                                 overwrite=FLAGS.overwrite,
                                 gin_config_files=None,
                                 gin_bindings=preproces_gin_bindings)
    
  if FLAGS.model == "vae":
    gin_file = "3d_shape_vae.gin"
  if FLAGS.model == "bvae":
    gin_file = "3d_shape_bvae.gin"
  if FLAGS.model == "fvae":
    gin_file = "3d_shape_fvae.gin"
  if FLAGS.model == "btcvae":
    gin_file = "3d_shape_btcvae.gin"
  if FLAGS.model == "annvae":
    gin_file = "3d_shape_annvae.gin"

  vae_gin_bindings = [
    "model.random_seed = %d" %(FLAGS.rng),
    "dataset.name = '%s'" %(FLAGS.dataset)
    ]
  vae_path = os.path.join(base_path, FLAGS.model + FLAGS.dataset + FLAGS.rng)
  train_vae_path = os.path.join(vae_path, 'model')
  unsupervised_train_partial.train_with_gin(
      train_vae_path, FLAGS.overwrite, [gin_file], vae_gin_bindings)

  postprocess_gin_bindings = [
      "postprocess.postprocess_fn = @mean_representation",
      "dataset.name='dummy_data'", 
      "postprocess.random_seed = %d"  %(FLAGS.rng)
      ]

  representation_path = os.path.join(vae_path, "representation")
  model_path = os.path.join(vae_path, "model")
  postprocess.postprocess_with_gin(
      model_path, representation_path, FLAGS.overwrite, gin_config_files=None, 
      gin_bindings=postprocess_gin_bindings)
  
  
  downstream_regression_train_gin_bindings = [
      "evaluation.evaluation_fn = @downstream_regression_on_representations",
      "dataset.name = '3dshapes_task'",
      "evaluation.random_seed = %d" %(FLAGS.rng),
      "downstream_regression_on_representations.num_train = [100]", #[127500]",
      "downstream_regression_on_representations.num_test = 50", #22500",
      "predictor.predictor_fn = @mlp_regressor",
      "mlp_regressor.hidden_layer_sizes = [32, 16]",
      "mlp_regressor.activation = 'logistic'",
      "mlp_regressor.max_iter = 100",
      "mlp_regressor.random_state = 0"
      ]
  
  result_path = os.path.join(vae_path, "metrics", "factor_regression")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, 
      gin_config_files=None, gin_bindings=downstream_regression_train_gin_bindings)
  
  downstream_reconstruction_train_gin_bindings = [
      "supervised_model.model = @downstream_decoder()",
      "supervised_model.batch_size = 64",
      "supervised_model.training_steps = 5", 
      "supervised_model.random_seed = %d" %(FLAGS.rng),
      "dataset.name='3dshapes_task'",
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
  
  result_path = os.path.join(vae_path, "metrics", "reconstruction")
  representation_path = os.path.join(vae_path, "representation")
  supervised_train_partial.train_with_gin(
      result_path, representation_path, FLAGS.overwrite,
      gin_bindings=downstream_reconstruction_train_gin_bindings)


if __name__ == "__main__":
  app.run(main)
