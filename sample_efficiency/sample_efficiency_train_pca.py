#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:54:26 2020

@author: petrapoklukar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import sys
sys.path.append('..')
from disentanglement_lib.preprocessing import preprocess
from disentanglement_lib.methods.unsupervised import pca

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")
flags.DEFINE_integer("rng", 0,
                     "random seed")


def main(unused_argv):
  base_path = "backbone"
  model_path = os.path.join(base_path, "pca")
  
  print("\n\n*- Preprocessing '%s' \n\n" %(FLAGS.dataset))
  preproces_gin_bindings = [
            "dataset.name = '%s'" %(FLAGS.dataset),
            "preprocess.preprocess_fn = @split_train_and_validation",
            "split_train_and_validation.random_seed = %d" %(FLAGS.rng)
      ]

  preprocess.preprocess_with_gin(FLAGS.dataset,
                                 "dummy_name",
                                 overwrite=FLAGS.overwrite,
                                 gin_config_files=None,
                                 gin_bindings=preproces_gin_bindings)
  print("\n\n*- Preprocessing DONE \n\n")
  
  print("\n\n*- Training PCA.")
  gin_bindings = [
      "dataset.name = '%s'" %(FLAGS.dataset),
      "train_pca.random_seed = 0",
      "train_pca.num_pca_components = [10, 30]",#[100, 500, 1000, 2000, 4000]",
  ]
  pca.train_pca_with_gin(
      model_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  
  preprocess.destroy_train_and_validation_splits(
      FLAGS.dataset + '_' + str(FLAGS.rng))
  print("\n\n*- Training COMPLETED \n\n")


if __name__ == "__main__":
  app.run(main)
