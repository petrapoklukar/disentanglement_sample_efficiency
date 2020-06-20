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
flags.DEFINE_boolean("save", True,
                     "Whether to save to file")
flags.DEFINE_boolean("agg", True,
                     "aggregate random seeds")

def main(unused_argv):

  assert FLAGS.base_path != None, "please enter basepath --base_path="
  base_path = FLAGS.base_path

  patterns=[]

  #regression tasks
  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")
  print(pattern)
  patterns.append(pattern)
  #reconstructions tasks
  pattern = os.path.join(base_path,"*/metrics/reconstruction/results/json/evaluate_holdout_results.json")
  patterns.append(pattern)
  #train performance 
  pattern = os.path.join(base_path,"*/model/results/json/train_results.json")
  patterns.append(pattern)
  
  task_id=0
  for pattern in patterns:

      all_data=_get(pattern)
      #print(list(all_data.columns.values))
      regression_results=[]
      model_names=[]
      dataset_names=[]
      #print(all_data)
      for i in range(all_data.shape[0]):
        #extract relevant information from jason
        name=all_data.loc[i,'path'].split('/')[1]
        
        dataset_name=name.split('_')[2] 
        if FLAGS.agg:
            model_name=name.split('_')[0]   
            rng_name=name.split('_')[3]
        else:
            model_name=model_name=name.split('_')[0] + "_" +name.split('_')[3]   
            rng_name=-1

        if task_id==0:
            performance=all_data.loc[i,'100:mean_holdout_mse']
            #performance=all_data.loc[i,'127500:mean_holdout_mse']
        if task_id==1:
            performance=all_data.loc[i,'reconstruction_loss']
        if task_id==2:
            performance=all_data.loc[i,'loss']
        regression_results.append((model_name,dataset_name,rng_name,float(performance)))
        model_names.append(model_name)
        dataset_names.append(dataset_name)

      #get all unique model names and datasets
      model_names = sorted(set(model_names))
      dataset_names = sorted(set(dataset_names), key=lambda x: int(x[1:]))
      
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
      plt.figure(task_id)
      for plot in plot_lst:
        x=np.arange(start=0, stop=len(dataset_names), step=1)
        v=np.array(plot[1])
        y=v[:,1].astype('float')
        e=v[:,2].astype('float')
        plt.errorbar(x, y, e,  capsize=3,label=plot[0])

      plt.legend(loc="lower left")
      if task_id==0:
        plt.ylabel('Mean Squared Error')
        plt.title('regression downstreamtasks')
        savename="downstream_regression"
      if task_id==1:
        plt.ylabel('L2 loss')
        plt.title('reconstruction downstreamtasks')
        savename="downstream_recon"
      if task_id==2:
        plt.ylabel('Loss')
        plt.title('training performance')
        savename="training"

       

      
      plt.xticks(x, dataset_names, size='small',rotation='vertical')
      plt.xlabel('Datasets')
      if FLAGS.save:
          plt.savefig("results/"+savename +".png")
      else:
          plt.show()

      #write latex table:
      if task_id==0:
        file = open("results/"+"regression_results_latex_table.txt","w") 
      if task_id==1:
        file = open("results/"+"reconstruction_results_latex_table.txt","w") 
      if task_id==2:
        file = open("results/"+"training_results_latex_table.txt","w") 
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
      file.close()

      task_id+=1


def get_model_data(pattern, result_str):
  """
    Example: 
      
    result_str = 'kl_loss'
  """
  all_data=_get(pattern)
  print(list(all_data.columns.values))
  regression_results=[]
  model_names=[]
  dataset_names=[]
  print(all_data)
  for i in range(all_data.shape[0]):
    #extract relevant information from json
    name=all_data.loc[i,'path'].split('/')[1]
    model_name=name.split('_')[0]
    dataset_name=name.split('_')[2]
    rng_name=name.split('_')[3]
    performance=all_data.loc[i, result_str]
    regression_results.append((model_name,dataset_name,rng_name,float(performance)))
    model_names.append(model_name)
    dataset_names.append(dataset_name)
  return regression_results, model_names, dataset_names


def plot_regression_agg_model_results(pattern, result_str, ylabel, task_name):
  regression_results, model_names, dataset_names = get_model_data(
    pattern, result_str)
  model_names = sorted(set(model_names))
  dataset_names = sorted(set(dataset_names), key=lambda x: int(x[1:]))
  print(dataset_names)
  
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


  #plot the results
  plt.figure(0, figsize=(10, 5))
  for plot in plot_lst:
    x=np.arange(start=0, stop=len(dataset_names), step=1)
    v=np.array(plot[1])
    y=v[:,1].astype('float')
    e=v[:,2].astype('float')
    plt.errorbar(x, y, e,  capsize=3,label=plot[0])

  plt.legend(loc="upper right")
  plt.ylabel(ylabel)
  name = 'results/agg_models_{0}_{1}.png'.format(task_name, result_str.split(':')[-1])
  plt.suptitle(task_name + ', ' + result_str.split(':')[-1])

  plt.xticks(x, dataset_names, size='small',rotation='vertical')
  plt.xlabel('Datasets')
  plt.subplots_adjust(wspace=0.5, hspace=0.3)
  plt.savefig(name)
  plt.clf()
  plt.close()



def plot_regression_mse_model_results_2(pattern, result_str, ylabel, task_name):
  regression_results, model_names, dataset_names = get_model_data(
    pattern, result_str)

  #get all unique model names and datasets
  model_names = sorted(set(model_names))
  dataset_names = sorted(set(dataset_names), key=lambda x: int(x[1:]))

  print("Found " + str(len(model_names))+ " models and " + str(len(dataset_names))+ " datasets")

  #agregate the results
  plot_lst=[]
  for name in model_names:
    values=[]
    for dataset in dataset_names:
      t_mse=[]
      for regression_result in regression_results:
        if regression_result[0]==name and regression_result[1] ==dataset:
          t_mse.append(regression_result[3]*6)
      t_mean_mse=np.mean(t_mse)
      t_mean_std=np.std(t_mse)
      values.append((dataset,t_mean_mse,t_mean_std, np.array(t_mse)))
    plot_lst.append((name,values))
    
  #plot the results
  plt.figure(0, figsize=(10, 5))
  for plot in plot_lst:
    x=np.arange(start=0, stop=len(dataset_names), step=1)
    v=np.array(plot[1])
    y=v[:,1].astype('float')
    e=v[:,2].astype('float')
    plt.errorbar(x, y, e,  capsize=3,label=plot[0])

  plt.legend(loc="upper right")
  plt.ylabel(ylabel)
  name = 'results/agg_models_sum_{0}_{1}.png'.format(task_name, result_str.split(':')[-1])
  plt.suptitle(task_name + ', ' + result_str.split(':')[-1])

  plt.xticks(x, dataset_names, size='small',rotation='vertical')
  plt.xlabel('Datasets')
  plt.subplots_adjust(wspace=0.5, hspace=0.3)
  plt.savefig(name)
  plt.clf()
  plt.close()
    
#def plot_regression_mse_model_results(unused_argv):
#  base_path = FLAGS.base_path
#  # Regression tasks
#  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")
##  result_str_list = ['127500:holdout_mse_factor_0', '127500:holdout_mse_factor_1',
##    '127500:holdout_mse_factor_2', '127500:holdout_mse_factor_3', '127500:holdout_mse_factor_4',
##    '127500:holdout_mse_factor_5'] 
##  agg_pds = []
##  for result_str in result_str_list:
##    regression_results, model_names, dataset_names = get_model_data(
##      pattern, result_str)
##    factor_regression_results = list(map(lambda x: [result_str] + list(x), regression_results))
##    agg_pds += factor_regression_results
##
##  agg_pd = pd.DataFrame(agg_pds, columns=['factor', 'model_name', 'dataset_name', 'seed', 'score'])
##  factor_sum_agg_pd = agg_pd.groupby(['model_name', 'dataset_name', 'seed'])['score'].apply(lambda x : x.astype(float).sum()).reset_index()
##  avg_per_seed_factor_sum_agg_pd = factor_sum_agg_pd.groupby(['model_name', 'dataset_name'])['score'].mean().reset_index()
#  
#  agg_pds = []
#  regression_results, model_names, dataset_names = get_model_data(
#      pattern, '127500:mean_holdout_mse')
#  factor_regression_results = list(map(lambda x: [x[0], x[1], x[2], x[3]*6], regression_results))
#  agg_pds += factor_regression_results
#  agg_pd = pd.DataFrame(agg_pds, columns=['model_name', 'dataset_name', 'seed', 'score'])
#  avg_per_seed_factor_sum_pd = agg_pd.groupby(['model_name', 'dataset_name'])['score'].mean().reset_index()
#  
#  print(avg_per_seed_factor_sum_pd)
#  print(avg_per_seed_factor_sum_pd.shape)
#  print(avg_per_seed_factor_sum_pd.dataset_name.shape)
#  dataset_names = sorted(list(avg_per_seed_factor_sum_pd.dataset_name.unique()),
#                         key=lambda x: int(x[1:]))
#  model_names = sorted(set(avg_per_seed_factor_sum_pd.model_name.unique()))
#  
#  plt.figure(0, figsize=(10, 5))
#  for model in model_names:
#    print(model)
#    df_model = avg_per_seed_factor_sum_pd.query("model_name == {0}".format(model))
#    print(df_model)
#    plt.plot(df_model.dataset_name, df_model.score)
#  plt.show()
##  x=np.arange(start=0, stop=len(dataset_names), step=1)
##    y=sorted(avg_per_seed_factor_sum_pd.query("model_name == {0}"), key=
##    v=np.array(plot_lst[plot][1])
##    y=v[:,1].astype('float')
##    e=v[:,2].astype('float')
##    print(x.repeat(3).shape, np.concatenate(v[:,3]).astype('float').shape)
##    plt.errorbar(x, y, e,  capsize=3, color='lightskyblue')
##    plt.scatter(x.repeat(3), np.concatenate(v[:,3]).astype('float'))
##    plt.title(plot_lst[plot][0])
##    plt.xticks([], [])
##    if plot >= 3:
##      plt.xticks(x, dataset_names, size='small',rotation='vertical')
##      plt.xlabel('Datasets')
##
##    if plot in [0, 3]:
##      plt.ylabel(ylabel)
##    name = 'results/per_model_{0}_{1}.png'.format(task_name, result_str.split(':')[-1])
##    plt.suptitle(task_name + ', ' + result_str.split(':')[-1])
##
##  plt.subplots_adjust(wspace=0.5, hspace=0.3)
##  plt.savefig(name)
##  plt.clf()
##  plt.close()


def plot_regression_model_results(pattern, result_str, ylabel, task_name):
  regression_results, model_names, dataset_names = get_model_data(
    pattern, result_str)

  #get all unique model names and datasets
  model_names = sorted(set(model_names))
  dataset_names = sorted(set(dataset_names), key=lambda x: int(x[1:]))

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
      values.append((dataset,t_mean_mse,t_mean_std, np.array(t_mse)))
    plot_lst.append((name,values))

  #plot the results
  plt.figure(0, figsize=(10, 5))
  for plot in range(len(plot_lst)):
    plt.subplot(2, 3, plot + 1)
    x=np.arange(start=0, stop=len(dataset_names), step=1)
    v=np.array(plot_lst[plot][1])
    y=v[:,1].astype('float')
    e=v[:,2].astype('float')
    print(x.repeat(3).shape, np.concatenate(v[:,3]).astype('float').shape)
    plt.errorbar(x, y, e,  capsize=3, color='lightskyblue')
    plt.scatter(x.repeat(3), np.concatenate(v[:,3]).astype('float'))
    plt.title(plot_lst[plot][0])
    plt.xticks([], [])
    if plot >= 3:
      plt.xticks(x, dataset_names, size='small',rotation='vertical')
      plt.xlabel('Datasets')

    if plot in [0, 3]:
      plt.ylabel(ylabel)
    name = 'results/per_model_{0}_{1}.png'.format(task_name, result_str.split(':')[-1])
    plt.suptitle(task_name + ', ' + result_str.split(':')[-1])

  plt.subplots_adjust(wspace=0.5, hspace=0.3)
  plt.savefig(name)
  plt.clf()
  plt.close()


def plot_reconstruction_model_results(pattern, result_str):
  regression_results, model_names, dataset_names = get_model_data(
    pattern, result_str)

  #get all unique model names and datasets
  model_names = sorted(set(model_names))
  dataset_names = sorted(set(dataset_names), key=lambda x: int(x[1:]))

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
      values.append((dataset,t_mean_mse,t_mean_std, np.array(t_mse)))
    plot_lst.append((name,values))

  #plot the results
  plt.figure(0, figsize=(10, 5))
  for plot in range(len(plot_lst)):
    plt.subplot(2, 3, plot + 1)
    x=np.arange(start=0, stop=len(dataset_names), step=1)
    v=np.array(plot_lst[plot][1])
    y=v[:,1].astype('float')
    e=v[:,2].astype('float')
    plt.errorbar(x, y, e,  capsize=3, color='lightskyblue')
    plt.scatter(x.repeat(3), np.concatenate(v[:,3]).astype('float'))
    plt.title(plot_lst[plot][0])
    plt.xticks([], [])
    if plot >= 3:
      plt.xticks(x, dataset_names, size='small',rotation='vertical')
      plt.xlabel('Datasets')

    if plot in [0, 3]:
      plt.ylabel('Loss')
    name = 'results/per_model_training_{0}.png'.format(result_str)
    plt.suptitle('training, ' + result_str)

  plt.subplots_adjust(wspace=0.5, hspace=0.3)
  plt.savefig(name)
  plt.clf()
  plt.close()


def plot_per_model_results(unused_argv):
  base_path = FLAGS.base_path
  
  # Regression tasks
  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")
  result_str_list = ['127500:mean_holdout_mse', '127500:holdout_mse_factor_0', '127500:holdout_mse_factor_1',
    '127500:holdout_mse_factor_2', '127500:holdout_mse_factor_3', '127500:holdout_mse_factor_4',
    '127500:holdout_mse_factor_5']
  for result_str in result_str_list:
    plot_regression_model_results(pattern, result_str, ylabel='Mean Squared Error', 
    task_name='regression_downstreamtasks')

  # Reconstruction tasks
  pattern = os.path.join(base_path,"*/metrics/reconstruction/results/json/evaluate_holdout_results.json")
  plot_regression_model_results(pattern, 'reconstruction_loss', ylabel='L2 Loss', 
    task_name='reconstruction_downstreamtask')

  # Train performance 
  pattern = os.path.join(base_path,"*/model/results/json/train_results.json")
  result_str_list = ['elbo', 'kl_loss', 'reconstruction_loss', 'loss', 'regularizer']
  for result_str in result_str_list:
    plot_reconstruction_model_results(pattern, result_str)


def plot_agg_models_results(unused_argv):
  base_path = FLAGS.base_path
  
  # Regression tasks
  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")
  result_str_list = ['127500:mean_holdout_mse', '127500:holdout_mse_factor_0', '127500:holdout_mse_factor_1',
    '127500:holdout_mse_factor_2', '127500:holdout_mse_factor_3', '127500:holdout_mse_factor_4',
    '127500:holdout_mse_factor_5']
  for result_str in result_str_list:
    plot_regression_agg_model_results(pattern, result_str, ylabel='Mean Squared Error', 
    task_name='regression_downstreamtasks')

  # Reconstruction tasks
  pattern = os.path.join(base_path,"*/metrics/reconstruction/results/json/evaluate_holdout_results.json")
  plot_regression_agg_model_results(pattern, 'reconstruction_loss', ylabel='L2 Loss', 
    task_name='reconstruction_downstreamtask')

  # Train performance 
  pattern = os.path.join(base_path,"*/model/results/json/train_results.json")
  result_str_list = ['elbo', 'kl_loss', 'reconstruction_loss', 'loss', 'regularizer']
  for result_str in result_str_list:
    plot_regression_agg_model_results(pattern, result_str, ylabel='Loss', 
    task_name='training')

def plot_agg_sum_models_results(unused_argv):
  base_path = FLAGS.base_path
  pattern = os.path.join(base_path,"*/metrics/factor_regression/results/json/evaluation_results.json")
  plot_regression_mse_model_results_2(pattern, '127500:mean_holdout_mse', ylabel='Mean Squared Error', 
    task_name='regression_downstreamtasks_sum')

def plot_agg_models_mig(unused_argv):
  base_path = FLAGS.base_path
  task_name = 'mig_10000'
  pattern = os.path.join(base_path,"*/metrics/{0}/results/json/evaluation_results.json".format(task_name))
  plot_regression_agg_model_results(pattern, 'discrete_mig', ylabel='MIG', 
    task_name=task_name)

def plot_agg_models_bvae(unused_argv):
  base_path = FLAGS.base_path
  task_name = 'bvae_10000'
  pattern = os.path.join(base_path,"*/metrics/{0}/results/json/evaluation_results.json".format(task_name))
  plot_regression_agg_model_results(pattern, 'eval_accuracy', ylabel='BetaVAE', 
    task_name=task_name)

def plot_agg_models_fvae(unused_argv):
  base_path = FLAGS.base_path
  task_name = 'fvae_10000'
  pattern = os.path.join(base_path,"*/metrics/{0}/results/json/evaluation_results.json".format(task_name))
  plot_regression_agg_model_results(pattern, 'eval_accuracy', ylabel='FactorVAE', 
    task_name=task_name)

def plot_agg_models_dci(unused_argv):
  task_name = 'dci_10000'
  base_path = FLAGS.base_path
  pattern = os.path.join(base_path,"*/metrics/{0}/results/json/evaluation_results.json".format(task_name))
  result_str_list = ['informativeness_test', 'disentanglement', 'completeness']
  for result_str in result_str_list:
    plot_regression_agg_model_results(pattern, result_str, ylabel='DCI', 
      task_name=task_name)

def run_all(unused_argv):
  app.run(plot_agg_models_mig)
  app.run(plot_agg_models_bvae)
  app.run(plot_agg_models_fvae)
  app.run(plot_agg_models_dci)
  # app.run(plot_agg_models_results)
  # app.run(plot_per_model_results)
  
if __name__ == "__main__":
  app.run(plot_agg_models_dci)
  # app.run(plot_agg_sum_models_results)
#  app.run(plot_per_model_results)
    #app.run(main)
