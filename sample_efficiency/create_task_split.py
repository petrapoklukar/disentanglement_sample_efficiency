#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:32:58 2020

@author: petrapoklukar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('..')
from disentanglement_lib.preprocessing import preprocess

preproces_task_gin_bindings = [
        "preprocess.preprocess_fn = @split_train_and_validation",
        "split_train_and_validation.random_seed = 0",
        "split_train_and_validation.unit_labels = True"
]

preprocess.preprocess_with_gin('3dshapes_task',
                               '',
                               overwrite=False,
                               gin_config_files=None,
                               gin_bindings=preproces_task_gin_bindings)
