#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:46:32 2020

@author: petrapoklukar
"""

from matplotlib import pyplot as plt
import numpy as np
import h5py

# load dataset
dataset = h5py.File('datasets/3dshapes/3dshapes.h5', 'r')
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
      

# methods for creating splits for the sample efficiency test
def create_top_splits():
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
      

def write_datasets(indices_list, savename_list, images, labels, indices):
    """ Creates a partial 3dshapes datasets from the given set of indices.
    Args:
        indices: list of indices in the original 3dshapes datasets
        name: name of the file to save the samples
    """
    print('Creating np arrays...')
    ims = np.array(images)
    print('Creating np arrays...')
    labs = np.array(labels)
    print('Creating np arrays...')
    inds = np.array(indices)
    
    print('Writing files')
    for indices, savename in list(zip(indices_list, savename_list)):
        hf = h5py.File('datasets/{0}.h5'.format(savename), 'w')
        hf.create_dataset('images', data=ims[indices])
        hf.create_dataset('labels', data=labs[indices])
        hf.create_dataset('indices', data=inds[indices])
        hf.close()


def plot_label_distribution(indices, labels, max_ind=10000):
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
    top_split_indices = create_top_splits()
    top_split_names = ['3dshapes_model_all', '3dshapes_task', '3dshapes_holdout']
    write_datasets(top_split_indices, top_split_names, images, labels)


def create_model_splits(filename):
    """ Randomly splits the model split into smaller datasets of different
        sizes.
    
    Args:   
        filename: name of the file to split further
    """
    dataset_split = h5py.File('datasets/{0}.h5'.format(filename), 'r')
    print(dataset_split.keys())
    images_split = dataset_split['images'] 
    labels_split = dataset_split['labels'] 
    indices_split = dataset_split['indices']
    
    print(len(images_split))
    np.random.seed(2610)
    all_indices = np.arange(len(indices_split))
    split_sizes = [10000, 50000, 100000, 150000, 250000]
    
    split_indices = []
    for split_size in split_sizes:
        model_split_indices = all_indices[:split_size]    
        assert(model_split_indices.shape[0] == split_size)
        split_indices.append(model_split_indices)
    
    assert(np.intersect1d(split_indices[0], 
                          split_indices[num_splits]).size == split_sizes[0] \
            for num_splits in range(len(split_sizes)))
    
    savename_list = []
    for split_size in range(len(split_sizes)): 
        savename_list.append('3dshapes_model_s{0}'.format(split_sizes[split_size]))
        
    write_datasets(split_indices, savename_list, images_split, labels_split, indices_split)
    dataset_split.close()


def check_factor_distribution(filename):
    """ Checks the distributions of generative factors in the given dataset split.
    
    Args:   
        filename: name of the dataset file to load
    """
    dataset_split = h5py.File('datasets/{0}.h5'.format(filename), 'r')
    print(dataset.keys())
    dataset_labels = dataset_split['labels']  
    dataset_indices = np.arange(len(dataset_labels))
    plot_label_distribution(dataset_indices, dataset_labels)


def create_split_train_and_validation(dataset_name, 
                                      random_state, 
                                      unit_labels=False):
    """ Randomly splits the model split into smaller datasets of different
        sizes.
    
    Args:   
        filename: name of the file to split further
    """
    SHAPES3D_PATH = 'datasets/3dshapes/{0}.h5'.format(dataset_name)
    dataset_split = h5py.File(SHAPES3D_PATH, 'r')
    print(dataset_split.keys())
    images_split = dataset_split['images'][()]
    labels_split = dataset_split['labels'][()]
    indices_split = dataset_split['indices'][()]
    dataset_size = len(images_split)
    
    ims = np.array(images_split)
    labs = np.array(labels_split)
    inds = np.array(indices_split)
    print(ims.shape, labs.shape, inds.shape)
    
    if unit_labels:
        labels_min = np.array([0., 0., 0., 0.75, 0., -30.])
        labels_max = np.array([0.9, 0.9, 0.9, 1.25, 3., 30.])
        labels_split = (labels_split - labels_min)/(labels_max - labels_min)
        print(labels_split.shape)
        assert(np.min(labels_split) == 0 and np.max(labels_split) == 1)
    
    print(dataset_size)
    all_local_indices = random_state.choice(dataset_size, dataset_size, replace=False)
    random_state.shuffle(all_local_indices)
    splitratio = int(dataset_size * 0.15)
    print(all_local_indices.shape)

    train_local_indices = all_local_indices[:splitratio]
    test_local_indices = all_local_indices[splitratio:]
    print(train_local_indices.shape, test_local_indices.shape)
    
    print('Writing files')
    for indices, split in list(zip([train_local_indices, test_local_indices], 
                                      ['_train', '_valid'])):
#        SPLIT_SHAPES3D_PATH = os.path.join(
#            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", 
#            dataset_name + split + ".h5")
        SPLIT_SHAPES3D_PATH = 'datasets/3dshapes/{0}{1}.h5'.format(dataset_name, split)

        assert(ims[indices].shape[0] == indices.shape[0])
        assert(labs[indices].shape[0] == indices.shape[0])
        assert(inds[indices].shape[0] == indices.shape[0])
        hf = h5py.File(SPLIT_SHAPES3D_PATH, 'w')
        hf.create_dataset('images', data=ims[indices])
        hf.create_dataset('labels', data=labs[indices])
        hf.create_dataset('indices', data=inds[indices])
        hf.close()
        
    dataset_split.close()

def main():
    create_top_datasets()
    create_model_splits('3dshapes_model_all')


if __name__ == '__main__':
    random_state = np.random.RandomState(234)
    create_split_train_and_validation('3dshapes_model_s10000', 
                                      random_state, 
                                      unit_labels=False)
    