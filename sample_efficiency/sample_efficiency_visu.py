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
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")

def main(unused_argv):

  datasets = ["3dshapes_model_s1000", "3dshapes_model_s10000", 
              "3dshapes_model_s50000", "3dshapes_model_s100000",
              "3dshapes_model_s150000", "3dshapes_model_s250000"]
  base_path = "3d_shape_out"
  
  
  for dataset in datasets:
    print(dataset)
    path =base_path+"/" + FLAGS.model+dataset
    visualize_model.visualize(path+"/model", path+"/vis", FLAGS.overwrite)


if __name__ == "__main__":
  app.run(main)
