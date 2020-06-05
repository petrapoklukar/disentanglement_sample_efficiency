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

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "vae model to use")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")
flags.DEFINE_integer("rng", 0,
                     "random seed")


def main(unused_argv):

  datasets=["3dshapes_model_s1000","3dshapes_model_s10000","3dshapes_model_s50000","3dshapes_model_s100000","3dshapes_model_s150000","3dshapes_model_s250000"]
  base_path="3d_shape_out"
  # petras spliting goes here:




  for dataset in datasets:
    if FLAGS.model=="vae":
      gin_file="3d_shape_vae.gin"
    if FLAGS.model=="bvae":
      gin_file="3d_shape_bvae.gin"
    if FLAGS.model=="fvae":
      gin_file="3d_shape_fvae.gin"
    if FLAGS.model=="btcvae":
      gin_file="3d_shape_btcvae.gin"
    if FLAGS.model=="annvae":
      gin_file="3d_shape_annvae.gin"



    gin_bindings = [
      "model.random_seed = %d" %(FLAGS.rng),
      "dataset.name = '%s'" %(dataset)
    ]
    model_path = os.path.join(base_path, FLAGS.model+dataset)
    train_partial.train_with_gin(model_path, FLAGS.overwrite,
                         [gin_file], gin_bindings)



if __name__ == "__main__":
  app.run(main)
