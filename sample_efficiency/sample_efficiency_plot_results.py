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
import multiprocessing
import simplejson as json
from tensorflow import gfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _get(pattern):
  files = gfile.Glob(pattern)
  pool = multiprocessing.Pool()
  all_results = pool.map(_load, files)
  return pd.DataFrame(all_results)

def _load(path):
  with gfile.GFile(path) as f:
    result = json.load(f)
  result["path"] = path
  return result


FLAGS = flags.FLAGS
flags.DEFINE_string("base_path", None, "base path with all the models")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")

def main(unused_argv):

  base_path = FLAGS.base_path

  #regression tasks
  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")

  print(pattern)

  all_data=_get(pattern)
  print(list(all_data.columns.values))
  regression_results=[]
  model_names=[]
  dataset_names=[]
  for i in range(all_data.shape[0]):
    #extract relevant information from jason
    name=all_data.loc[i,'path'].split('/')[1]
    model_name=name.split('_')[0]
    dataset_name=name.split('_')[2]
    rng_name=name.split('_')[3]
    mean_holdout_mse=all_data.loc[i,'100:mean_holdout_mse']
    regression_results.append((model_name,dataset_name,rng_name,float(mean_holdout_mse)))
    model_names.append(model_name)
    dataset_names.append(dataset_name)

  #get all unique model names and datasets
  model_names = set(model_names)
  dataset_names = set(dataset_names)

  print("Found " + str(len(model_names))+ " models and " + str(len(dataset_names))+ " datasets")

  #agregate the results
  plot_lst=[]
  for name in model_names:
    values=[]
    for dataset in dataset_names:
      t_mse=[]
      for regression_result in regression_results:
        if regression_result[0]==name and regression_result[1] ==dataset:
          t_mse.append(regression_result[3])
      t_mean_mse=np.mean(t_mse)
      t_mean_std=np.std(t_mse)
      values.append((dataset,t_mean_mse,t_mean_std))
    plot_lst.append((name,values))

  #plot the 
  for plot in plot_lst:
    x=np.arange(start=0, stop=len(dataset_names), step=1)
    v=np.array(plot[1])
    y=v[:,1].astype('float')
    e=v[:,2].astype('float')
    plt.errorbar(x, y, e,  capsize=3,label=plot[0])

  plt.legend(loc="lower left")
  plt.ylabel('disentagelment metric')
  plt.xticks(x, dataset_names, size='small',rotation='vertical')
  plt.xlabel('model checkpoints')
  plt.show()

  #write latex table:
  file = open("regression_results_latex_table.txt","w") 
  file.write("Model &") 
  for dataset_name in dataset_names:
      file.write(str(dataset_name)) 
      file.write(" & ") 
  file.write("\n")
  for plot in plot_lst:
    file.write(str(plot[0])+" & ") 
    v=np.array(plot[1])
    y=v[:,1].astype('float')
    std=v[:,2].astype('float')
    for py,pstd in zip(y,std):
      file.write("$" + str(np.round(py,2))+" \\pm " + str(np.round(pstd,2))+"$ & ")
    file.write("\n")




  
  


if __name__ == "__main__":
  app.run(main)
