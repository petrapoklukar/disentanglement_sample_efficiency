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
from disentanglement_lib.methods.unsupervised import train_partial
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

  vae_path = os.path.join(base_path, FLAGS.model + FLAGS.dataset + '_' + str(FLAGS.rng))
  train_vae_path = os.path.join(vae_path, 'model')
  visualize_model.visualize(train_vae_path, vae_path + "/vis", FLAGS.overwrite)
  preprocess.destroy_train_and_validation_splits(FLAGS.dataset + '_' + FLAGS.model + '_' + str(FLAGS.rng))

if __name__ == "__main__":
  app.run(main)
