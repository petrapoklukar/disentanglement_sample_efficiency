#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:47:52 2020

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
from disentanglement_lib.preprocessing import preprocess
from disentanglement_lib.evaluation import evaluate_with_decodings

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
#  vae_path = os.path.join(base_path, FLAGS.model)
  model_path = os.path.join(vae_path, "model")
  print(vae_path, model_path)
  
  print("\n\n*- Preprocessing '%s' \n\n" %(FLAGS.dataset))
  preproces_gin_bindings = [
            "dataset.name = '%s'" %(FLAGS.dataset),
            "preprocess.preprocess_fn = @split_train_and_validation",
            "split_train_and_validation.random_seed = %d" %(FLAGS.rng)
      ]

  preprocess.preprocess_with_gin(FLAGS.dataset,
                                 FLAGS.model,
                                 overwrite=FLAGS.overwrite,
                                 gin_config_files=None,
                                 gin_bindings=preproces_gin_bindings)
  print("\n\n*- Preprocessing DONE \n\n")

  print("\n\n*- Evaluating Recall.")
  gin_bindings = [
      "evaluate_with_decodings.evaluation_fn = @recall",
      "evaluate_with_decodings.postprocess_fn = @mean_representation",
      "evaluate_with_decodings.random_seed = 0",
      "dataset.name='3dshapes_model_s1000'",
      "recall.num_recall_samples = 100",
      "recall.nhood_sizes = [3, 5]",
      "recall.num_interventions_per_latent_dim = 5"
  ]
  result_path = os.path.join(vae_path, "metrics", "test_recall_100")
  evaluate_with_decodings.evaluate_with_gin(
      model_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  preprocess.destroy_train_and_validation_splits(
      FLAGS.dataset + '_' + FLAGS.model + '_' + str(FLAGS.rng))
  print("\n\n*- Evaluation COMPLETED \n\n")


if __name__ == "__main__":
  app.run(main)
