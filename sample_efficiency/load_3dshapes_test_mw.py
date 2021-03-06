#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:46:32 2020

@author: petrapoklukar
"""

from matplotlib import pyplot as plt
import numpy as np
import h5py


# methods for creating splits for the sample efficiency test
def create_top_splits(images):
    """ Randomly splits the entire 3dshapes dataset into model split, 
        task split and holdout split.
    
    Returns:
        model_indices: list of length model_split
        task_indices: list of length task_split
        holdout_indices: list of length holdout_split
    """
    np.random.seed(2610)
    all_indices = np.arange(len(images))
    np.random.shuffle(all_indices)

    model_split = 250000
    task_split = 150000
    holdout_split = 80000
    model_indices = all_indices[:model_split]
    task_indices = all_indices[model_split:model_split + task_split]
    holdout_indices = all_indices[model_split + task_split:]
    assert(model_indices.shape[0] == model_split)
    assert(task_indices.shape[0] == task_split)
    assert(holdout_indices.shape[0] == holdout_split)
    assert(not np.intersect1d(model_indices, task_indices).size > 0)
    assert(not np.intersect1d(task_indices, holdout_indices).size > 0)
    return model_indices, task_indices, holdout_indices




# load dataset
dataset = h5py.File('datasets/3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
labels = dataset['labels']  # array shape [480000,6], float64
image_shape = images.shape[1:]  # [64,64,3]
label_shape = labels.shape[1:]  # [6]
n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}


print(images.shape)


images_np=images[:480000]
print(images_np.shape)
model_indices, task_indices, holdout_indices = create_top_splits(images)
print(holdout_indices.shape)
images_np_sel=images_np[holdout_indices]
lables_np_=images_np[holdout_indices]
print(images_np_sel.shape)




# methods for sampling unconditionally/conditionally on a given factor
def get_index(factors):
  """ Converts factors to indices in range(num_data)
  Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

  Returns:
    indices: np array shape [batch_size].
  """
  indices = 0
  base = 1
  for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
      print(factor, name)
      print(indices, base)
      indices += factors[factor] * base
      base *= _NUM_VALUES_PER_FACTOR[name]
      print(indices, base)
  return indices


def sample_random_batch(batch_size):
  """ Samples a random batch of images.
  Args:
    batch_size: number of images to sample.

  Returns:
    batch: images shape [batch_size,64,64,3].
  """
  indices = np.random.choice(n_samples, batch_size)
  ims = []
  labs = []
  for ind in indices:
    im = images[ind]
    im = np.asarray(im)
    ims.append(im)
    lab = labels[ind]
    lab = np.asarray(lab)
    labs.append(lab)
  ims = np.stack(ims, axis=0)
  ims = ims / 255. # normalise values to range [0,1]
  ims = ims.astype(np.float32)
  return ims.reshape([batch_size, 64, 64, 3]), labs


def sample_batch(batch_size, fixed_factor, fixed_factor_value):
  """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
      the other factors varying randomly.
  Args:
    batch_size: number of images to sample.
    fixed_factor: index of factor that is fixed in range(6).
    fixed_factor_value: integer value of factor that is fixed 
      in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

  Returns:
    batch: images shape [batch_size,64,64,3]
  """
  factors = np.zeros([len(_FACTORS_IN_ORDER), batch_size],
                     dtype=np.int32)
  for factor, name in enumerate(_FACTORS_IN_ORDER):
    num_choices = _NUM_VALUES_PER_FACTOR[name]
    factors[factor] = np.random.choice(num_choices, batch_size)
  factors[fixed_factor] = fixed_factor_value
  indices = get_index(factors)
  ims = []
  for ind in indices:
    im = images[ind]
    im = np.asarray(im)
    ims.append(im)
  ims = np.stack(ims, axis=0)
  ims = ims / 255. # normalise values to range [0,1]
  ims = ims.astype(np.float32)
  return ims.reshape([batch_size, 64, 64, 3])


def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')
      


    
    
def create_dataset(indices, savename, images, labels):
    """ Creates a partial 3dshapes datasets from the given set of indices.
    Args:
        indices: list of indices in the original 3dshapes datasets
        name: name of the file to save the samples
    """
    ims, labs = [], []
    counter = 0
    dataset_len = len(indices) 
    print("lets go")
    for ind in indices:
        image = np.asarray(images[ind])
        ims.append(image)
        label = np.asarray(labels[ind])
        labs.append(label)
        counter += 1 
        if counter % 1 == 0:
            print('Processed {0}/{1}'.format(counter, dataset_len))
        
    hf = h5py.File('datasets/{0}.h5'.format(savename), 'w')
    hf.create_dataset('images', data=ims)
    hf.create_dataset('labels', data=labs)
    hf.create_dataset('indices', data=indices)
    hf.close()


def plot_label_distribution(indices, max_ind=10000):
    """ Plots the distribution over generative factors obtained from the labels
        list given the list of indices.
    Args:
        indices: list of indices corresponding to samples in the original
            3dshapes datasets
        max_ind: limit the set of indices to consider in the plot
    """
    labels_plot = np.stack([labels[j] for j in indices[:max_ind]])
    for i in range(len(_FACTORS_IN_ORDER)):    
        plt.figure(1 + i)
        plt.clf()
        plt.hist(labels_plot[:, i], label=_FACTORS_IN_ORDER[i], 
                 bins=_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]])
        plt.show()
        
    
def create_top_datasets():    
    """ Creates random model split, task split and holdout splits. The splits
        are disjoint.
    """
    model_indices, task_indices, holdout_indices = create_top_splits()
    create_dataset(model_indices, '3dshapes_model_all', images, labels)
    create_dataset(task_indices, '3dshapes_task', images, labels)
    create_dataset(holdout_indices, '3dshapes_holdout', images, labels)
    

def create_model_splits(filename, savename):
    """ Randomly splits the model split into smaller datasets of different
        sizes.
    
    Returns:

    """
    dataset_split = h5py.File('datasets/{0}.h5'.format(filename), 'r')
    print(dataset_split.keys())
    images_split = dataset_split['images'] 
    labels_split = dataset_split['labels'] 
    np.random.seed(2610)
    all_indices = np.arange(len(images))
    split_sizes = [10000, 50000, 100000, 150000, 250000]
    
    split_indices = []
    for split_size in split_sizes:
        model_split_indices = all_indices[:split_size]    
        assert(model_split_indices.shape[0] == split_size)
        split_indices.append(model_split_indices)
    
    assert(np.intersect1d(split_indices[0], 
                          split_indices[num_splits]).size == split_sizes[0] \
            for num_splits in range(len(split_size)))
    
    for split_size in range(len(split_sizes)): 
        savename = '3dshapes_model_s{0}'.format(split_sizes[split_size])
        create_dataset(split_indices[split_size], savename, 
                       images_split, labels_split)       
    dataset_split.close()




#img_batch = sample_batch(batch_size, 5, 15)
#show_images_grid(img_batch, num_images=16)

#hf = h5py.File('datasets/3dshapes_test.h5', 'w')
#hf.create_dataset('images', data=random_batch[0])
#hf.create_dataset('labels', data=random_batch[1])
#hf.close()
#
#dataset_test = h5py.File('datasets/3dshapes_test.h5', 'r')
#dataset_test.close()
