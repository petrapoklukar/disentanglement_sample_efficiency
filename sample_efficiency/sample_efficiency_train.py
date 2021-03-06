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
import time
from absl import flags
import sys
sys.path.append('..')
from disentanglement_lib.methods.unsupervised import train_partial as unsupervised_train_partial
from disentanglement_lib.methods.supervised import train_partial as supervised_train_partial
from disentanglement_lib.preprocessing import preprocess
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.visualize import visualize_model

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "vae model to use")
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")
flags.DEFINE_integer("rng", 0,
                     "random seed")


def main(unused_argv):
  base_path = "3dshapes_models"

  done = False
  while not done:
    try:  
      print("\n\n*- Preprocessing '%s' \n\n" %(FLAGS.dataset))
      preproces_gin_bindings = [
            "dataset.name = '%s'" %(FLAGS.dataset),
            "preprocess.preprocess_fn = @split_train_and_validation_per_model",
            "split_train_and_validation_per_model.random_seed = %d" %(FLAGS.rng)
      ]

      preprocess.preprocess_with_gin(FLAGS.dataset,
                                     FLAGS.model,
                                     overwrite=FLAGS.overwrite,
                                     gin_config_files=None,
                                     gin_bindings=preproces_gin_bindings)
      print("\n\n*- Preprocessing DONE \n\n")
      done = True
    except:
      time.sleep(30)

  if FLAGS.model == "vae":
    gin_file = "3d_shape_vae.gin"
  if FLAGS.model == "bvae":
    gin_file = "3d_shape_bvae.gin"
  if FLAGS.model == "b8vae":
    gin_file = "3d_shape_b8vae.gin"
  if FLAGS.model == "fvae":
    gin_file = "3d_shape_fvae.gin"
  if FLAGS.model == "btcvae":
    gin_file = "3d_shape_btcvae.gin"
  if FLAGS.model == "annvae":
    gin_file = "3d_shape_annvae.gin"
  if FLAGS.model == "randomvae":
    gin_file = "3d_shape_randomvae.gin"

  print("\n\n*- Training '%s' \n\n" %(FLAGS.model))
  vae_gin_bindings = [
    "model.random_seed = %d" %(FLAGS.rng),
    "dataset.name = '%s'" %(FLAGS.dataset + '_' + FLAGS.model + '_' + str(FLAGS.rng))
    ]
  vae_path = os.path.join(base_path, FLAGS.model + FLAGS.dataset + '_' + str(FLAGS.rng))
  train_vae_path = os.path.join(vae_path, 'model')
  unsupervised_train_partial.train_with_gin(
      train_vae_path, FLAGS.overwrite, [gin_file], vae_gin_bindings)
  visualize_model.visualize(train_vae_path, vae_path + "/vis", FLAGS.overwrite)
  preprocess.destroy_train_and_validation_splits(FLAGS.dataset + '_' + FLAGS.model + '_' + str(FLAGS.rng))
  print("\n\n*- Training DONE \n\n")

  print("\n\n*- Postprocessing '%s' \n\n" %(FLAGS.model))
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
  print("\n\n*- Postprocessing DONE \n\n")

  # --- Evaluate disentanglement metrics
  print("\n\n*- Evaluating MIG.")
  gin_bindings = [
     "evaluation.evaluation_fn = @mig",
     "dataset.name='3dshapes'",
     "evaluation.random_seed = 0",
     "mig.num_train = 10000",
     "discretizer.discretizer_fn = @histogram_discretizer",
     "discretizer.num_bins = 20"
  ]
  result_path = os.path.join(vae_path, "metrics", "mig")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating BetaVEA.")
  gin_bindings = [
      "evaluation.evaluation_fn = @beta_vae_sklearn",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "beta_vae_sklearn.batch_size = 16",
      "beta_vae_sklearn.num_train = 10000",
      "beta_vae_sklearn.num_eval = 5000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]
  result_path = os.path.join(vae_path, "metrics", "bvae")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating FactorVAE.")
  gin_bindings = [
      "evaluation.evaluation_fn = @factor_vae_score",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "factor_vae_score.batch_size = 16",
      "factor_vae_score.num_train = 10000",
      "factor_vae_score.num_eval = 5000",
      "factor_vae_score.num_variance_estimate = 10000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]
  
  result_path = os.path.join(vae_path, "metrics", "fvae")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating DCI.")
  gin_bindings = [
      "evaluation.evaluation_fn = @dci",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "dci.batch_size = 16",
      "dci.num_train = 10000",
      "dci.num_test = 5000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]

  result_path = os.path.join(vae_path, "metrics", "dci")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  print("\n\n*- Evaluation COMPLETED \n\n")

  # --- Downstream tasks
  print("\n\n*- Training downstream factor regression '%s' \n\n" %(FLAGS.model))
  downstream_regression_train_gin_bindings = [
      "evaluation.evaluation_fn = @downstream_regression_on_representations",
      "dataset.name = '3dshapes_task'",
      "evaluation.random_seed = 0",
      "downstream_regression_on_representations.holdout_dataset_name = '3dshapes_holdout'",
      "downstream_regression_on_representations.num_train = [127500]",
      "downstream_regression_on_representations.num_test = 22500",
      "downstream_regression_on_representations.num_holdout = 80000", 
      "predictor.predictor_fn = @mlp_regressor",
      "mlp_regressor.hidden_layer_sizes = [16, 8]",
      "mlp_regressor.activation = 'logistic'",
      "mlp_regressor.max_iter = 50",
      "mlp_regressor.random_state = 0"
      ]
  
  result_path = os.path.join(vae_path, "metrics", "factor_regression")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, 
      gin_config_files=None, gin_bindings=downstream_regression_train_gin_bindings)
  print("\n\n*- Training downstream factor regression DONE \n\n")
  
  print("\n\n*- Training downstream reconstruction '%s' \n\n" %(FLAGS.model))
  downstream_reconstruction_train_gin_bindings = [
      "supervised_model.model = @downstream_decoder()",
      "supervised_model.batch_size = 64",
      "supervised_model.training_steps = 30000", 
      "supervised_model.eval_steps = 1000", 
      "supervised_model.random_seed = 0",
      "supervised_model.holdout_dataset_name = '3dshapes_holdout'",
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
  supervised_train_partial.train_with_gin(
      result_path, representation_path, FLAGS.overwrite,
      gin_bindings=downstream_reconstruction_train_gin_bindings)
  visualize_model.visualize_supervised(result_path, representation_path, 
                                       result_path + "/vis", FLAGS.overwrite)

  print("\n\n*- Training downstream reconstruction DONE \n\n")
  print("\n\n*- Training & evaluation COMPLETED \n\n")


if __name__ == "__main__":
  app.run(main)
