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
from disentanglement_lib.methods.unsupervised import train_partial
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
import gin.tf

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "3d_shape_out"

# By default, we do not overwrite output directories. Set this to True, if you
# want to overwrite (in particular, if you rerun this script several times).
overwrite = True

# 1. Train a standard VAE (already implemented in disentanglement_lib).
# ------------------------------------------------------------------------------

# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "vae")
#train_partial.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["3d_shape_vae.gin"])


# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.
for path in [path_vae]:
  representation_path = os.path.join(path, "representation")
  model_path = os.path.join(path, "model")
  postprocess_gin = ["postprocess.gin"]  # This contains the settings.
  # postprocess.postprocess_with_gin defines the standard extraction protocol.
  postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                   postprocess_gin)


# 4. Train a downstream task
gin_bindings = [
    "evaluation.evaluation_fn = @downstream_task_on_representations",
    "dataset.name='3dshapes_task'",
    "evaluation.random_seed = 0",
    "downstream_task_on_representations.num_train=[100]",
    "downstream_task_on_representations.num_test=50",
    "predictor.predictor_fn = @mlp_regressor",
    "mlp_regressor.hidden_layer_sizes = [32, 16]",
    "mlp_regressor.activation='identity'",
    "mlp_regressor.max_iter=10",
    "mlp_regressor.random_state=0"
]
for path in [path_vae]:
  result_path = os.path.join(path, "metrics", "downstream_task")
  representation_path = os.path.join(path, "representation")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=gin_bindings)#["3d_shape_classifier.gin"])#gin_bindings=gin_bindings)

pa = 1/0
# 4. Compute the Mutual Information Gap (already implemented) for both models.
# ------------------------------------------------------------------------------
# The main evaluation protocol of disentanglement_lib is defined in the
# disentanglement_lib.evaluation.evaluate module. Again, we have to provide a
# gin configuration. We could define a .gin config file; however, in this case
# we show how all the configuration settings can be set using gin bindings.
# We use the Mutual Information Gap (with a low number of samples to make it
# faster). To learn more, have a look at the different scores in
# disentanglement_lib.evaluation.evaluate.metrics and the predefined .gin
# configuration files in
# disentanglement_lib/config/unsupervised_study_v1/metrics_configs/(...).
gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='3dshapes'",
    "evaluation.random_seed = 0",
    "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]
for path in [path_vae, path_bvae]:
  result_path = os.path.join(path, "metrics", "mig")
  representation_path = os.path.join(path, "representation")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=gin_bindings)


# 5. Compute a custom disentanglement metric for both models.
# ------------------------------------------------------------------------------
# The following function implements a dummy metric. Note that all metrics get
# ground_truth_data, representation_function, random_state arguments by the
# evaluation protocol, while all other arguments have to be configured via gin.
@gin.configurable(
    "custom_metric",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_custom_metric(ground_truth_data,
                          representation_function,
                          random_state,
                          num_train=gin.REQUIRED,
                          batch_size=16):
  """Example of a custom (dummy) metric.

  Preimplemented metrics can be found in disentanglement_lib.evaluation.metrics.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with disentanglement score.
  """
  score_dict = {}

  # This is how to obtain the representations of num_train points along with the
  # ground-truth factors of variation.
  representation, factors_of_variations = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train, random_state,
      batch_size)
  # We could now compute a metric based on representation and
  # factors_of_variations. However, for the sake of brevity, we just return 1.
  del representation, factors_of_variations
  score_dict["custom_metric"] = 1.
  return score_dict


# To compute the score, we again call the evaluation protocol with a gin
# configuration. At this point, note that for all steps, we have to set a
# random seed (in this case via `evaluation.random_seed`).
gin_bindings = [
    "evaluation.evaluation_fn = @custom_metric",
    "custom_metric.num_train = 100", "evaluation.random_seed = 0",
    "dataset.name='auto'"
]
for path in [path_vae, path_custom_vae]:
  result_path = os.path.join(path, "metrics", "custom_metric")
  evaluate.evaluate_with_gin(
      representation_path, result_path, overwrite, gin_bindings=gin_bindings)

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(
    pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)
