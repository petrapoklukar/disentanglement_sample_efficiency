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

"""Downstream classification task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
from six.moves import range
import gin.tf
from sklearn.metrics import mean_squared_error


@gin.configurable(
    "downstream_task",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_downstream_task(ground_truth_data,
                            representation_function,
                            random_state,
                            artifact_dir=None,
                            num_train=gin.REQUIRED,
                            num_test=gin.REQUIRED,
                            batch_size=16):
  """Computes loss of downstream task.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  del artifact_dir
  scores = {}
  for train_size in num_train:
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, train_size, random_state,
        batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test, random_state,
        batch_size)
    predictor_model = utils.make_predictor_fn()

    print(mus_train.shape, ys_train.shape)
    train_err, test_err = _compute_loss(
        np.transpose(mus_train), ys_train, np.transpose(mus_test),
        ys_test, predictor_model)
    size_string = str(train_size)
    scores[size_string +
           ":mean_train_accuracy"] = np.mean(train_err)
    scores[size_string +
           ":mean_test_accuracy"] = np.mean(test_err)
    scores[size_string +
           ":min_train_accuracy"] = np.min(train_err)
    scores[size_string + ":min_test_accuracy"] = np.min(test_err)
    for i in range(len(train_err)):
      scores[size_string +
             ":train_accuracy_factor_{}".format(i)] = train_err[i]
      scores[size_string + ":test_accuracy_factor_{}".format(i)] = test_err[i]

  return scores


@gin.configurable(
    "downstream_regression_on_representations",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_downstream_regression_on_representations(ground_truth_data,
                                                     ground_truth_holdout_data,
                                                     representation_function,
                                                     random_state,
                                                     artifact_dir=None,
                                                     num_train=gin.REQUIRED,
                                                     num_test=gin.REQUIRED,
                                                     num_holdout=gin.REQUIRED,
                                                     batch_size=16):
  """Computes loss of downstream task on representations.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  del artifact_dir
  ground_truth_train_data, ground_truth_test_data = ground_truth_data
  scores = {}
  for train_size in num_train:
    mus_train, ys_train = utils.generate_batch_label_code(
        ground_truth_train_data, representation_function, train_size, 
        random_state, batch_size)
    mus_test, ys_test = utils.generate_batch_label_code(
        ground_truth_test_data, representation_function, num_test, 
        random_state, batch_size)
    mus_holdout, ys_holdout = utils.generate_batch_label_code(
        ground_truth_holdout_data, representation_function, num_holdout, 
        random_state, batch_size)
    
    predictor_model = utils.make_predictor_fn()

    train_err, test_err, holdout_err, random_normal_err, random_uniform_err = \
      _compute_mse_loss(
        np.transpose(mus_train), ys_train, np.transpose(mus_test),
        ys_test, np.transpose(mus_holdout), ys_holdout, predictor_model)
    size_string = str(train_size)
    scores[size_string + ":mean_train_mse"] = np.mean(train_err)
    scores[size_string + ":mean_test_mse"] = np.mean(test_err)
    scores[size_string + ":mean_holdout_mse"] = np.mean(holdout_err)
    scores[size_string + ":mean_random_normal_mse"] = np.mean(random_normal_err)
    scores[size_string + ":mean_random_uniform_mse"] = np.mean(random_uniform_err)
    scores[size_string + ":min_train_mse"] = np.min(train_err)
    scores[size_string + ":min_test_mse"] = np.min(test_err)
    scores[size_string + ":min_holdout_mse"] = np.min(holdout_err)
    scores[size_string + ":min_random_normal_mse"] = np.min(random_normal_err)
    scores[size_string + ":min_random_uniform_mse"] = np.min(random_uniform_err)
    for i in range(len(train_err)):
      scores[size_string +
             ":train_mse_factor_{}".format(i)] = train_err[i]
      scores[size_string + ":test_mse_factor_{}".format(i)] = test_err[i]
      scores[size_string + ":holdout_mse_factor_{}".format(i)] = holdout_err[i]
      scores[size_string + ":random_normal_mse_factor_{}".format(i)] = random_normal_err[i]
      scores[size_string + ":random_uniform_mse_factor_{}".format(i)] = random_uniform_err[i]
  return scores


def _compute_loss(x_train, y_train, x_test, y_test, predictor_fn):
  """Compute average accuracy for train and test set."""
  num_factors = y_train.shape[0]
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = predictor_fn()
    model.fit(x_train, y_train[i, :])
    train_loss.append(np.mean(model.predict(x_train) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
  return train_loss, test_loss


def _compute_mse_loss(x_train, y_train, x_test, y_test, x_holdout, y_holdout, 
                      predictor_fn):
  """Compute average accuracy for train and test set."""
  num_factors = y_train.shape[0]
  train_loss = []
  test_loss = []
  holdout_loss = []
  random_normal_loss, random_uniform_loss = [], []
  x_random_normal = np.random.normal(size=x_train.shape)
  x_random_uniform = np.random.choice(
      np.concatenate([np.random.uniform(low=3, high=5, size=int(x_holdout.size/2)), 
                      np.random.uniform(low=-5, high=-3, size=int(x_holdout.size/2))]), 
      size=x_holdout.shape, 
      replace=False)
  for i in range(num_factors):
    model = predictor_fn()
    model.fit(x_train, y_train[i, :])
    train_loss.append(mean_squared_error(model.predict(x_train), y_train[i, :]))
    test_loss.append(mean_squared_error(model.predict(x_test), y_test[i, :]))
    holdout_loss.append(mean_squared_error(model.predict(x_holdout), y_holdout[i, :]))
    random_normal_loss.append(mean_squared_error(model.predict(x_random_normal), y_holdout[i, :]))
    random_uniform_loss.append(mean_squared_error(model.predict(x_random_uniform), y_holdout[i, :]))
  return train_loss, test_loss, holdout_loss, random_normal_loss, random_uniform_loss