#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:47:52 2020

@author: petrapoklukar
"""

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
  model_path = os.path.join(vae_path, "model")
  print(vae_path, model_path)

  print("\n\n*- Evaluating Recall.")
  gin_bindings = [
      "evaluate_with_decodings.evaluation_fn = @recall",
      "evaluate_with_decodings.postprocess_fn = @mean_representation",
      "evaluation.random_seed = 0",
      "dataset.name='3dshapes'",
      "recall.num_recall_samples = 100",
  ]
  result_path = os.path.join(vae_path, "metrics", "test_recall_100")
  evaluate_with_decodings.evaluate_with_gin(
      model_path, result_path, FLAGS.overwrite, gin_bindings=gin_bindings)
  print("\n\n*- Evaluation COMPLETED \n\n")


if __name__ == "__main__":
  app.run(main)
