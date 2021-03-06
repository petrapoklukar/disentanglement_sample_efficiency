#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:39:55 2020

@author: petrapoklukar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
import time
from absl import flags
import sys
sys.path.append('..')
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "vae model to use")
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")
flags.DEFINE_integer("rng", 0,
                     "random seed")


def main(unused_argv):
  base_path = "3dshapes_models"
  
  print("\n\n*- Evaluating '%s' \n\n" %(FLAGS.model))
  vae_path = os.path.join(base_path, FLAGS.model + FLAGS.dataset + '_' + str(FLAGS.rng))
  representation_path = os.path.join(vae_path, "representation")
  print(vae_path, representation_path)

  print("\n\n*- Evaluating MIG.")
  gin_bindings = [
     "evaluation.evaluation_fn = @mig",
     "dataset.name='3dshapes'",
     "evaluation.random_seed = 0",
     "mig.num_train = 100000",
     "discretizer.discretizer_fn = @histogram_discretizer",
     "discretizer.num_bins = 20"
  ]
  result_path = os.path.join(vae_path, "metrics", "mig_10000")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating BetaVEA.")
  gin_bindings = [
      "evaluation.evaluation_fn = @beta_vae_sklearn",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "beta_vae_sklearn.batch_size = 16",
      "beta_vae_sklearn.num_train = 100000",
      "beta_vae_sklearn.num_eval = 5000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]
  result_path = os.path.join(vae_path, "metrics", "bvae_10000")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating FactorVAE.")
  gin_bindings = [
      "evaluation.evaluation_fn = @factor_vae_score",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "factor_vae_score.batch_size = 16",
      "factor_vae_score.num_train = 100000",
      "factor_vae_score.num_eval = 5000",
      "factor_vae_score.num_variance_estimate = 100000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]
  
  result_path = os.path.join(vae_path, "metrics", "fvae_10000")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  print("\n\n*- Evaluating DCI.")
  gin_bindings = [
      "evaluation.evaluation_fn = @dci",
      "dataset.name='3dshapes'",
      "evaluation.random_seed = 0",
      "dci.batch_size = 16",
      "dci.num_train = 100000",
      "dci.num_test = 5000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
      ]

  result_path = os.path.join(vae_path, "metrics", "dci_10000")
  evaluate.evaluate_with_gin(
      representation_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)

  print("\n\n*- Evaluation COMPLETED \n\n")


if __name__ == "__main__":
  app.run(main)
