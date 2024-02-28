#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
/**

* © Copyright (C) 2016-2020 Xilinx, Inc
*
* Licensed under the Apache License, Versio[[n 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
'''

# modified by daniele.bagni@xilinx.com
# date 20 / 11 / 2020

import cv2
import os
import numpy as np
import pandas as pd

## Silence TensorFlow messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#from tensorflow.keras.preprocessing.image import img_to_array
from config import net_config as cfg
from config import fcn8_cnn as cnn
import data_utils as dus
import eval_utils as eus

DATAS_DIR     = cfg.DATASET_DIR
IMG_calib_DIR  = cfg.dir_calib_img
SEG_calib_DIR  = cfg.dir_calib_seg

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
BATCH_SIZE = cfg.BATCH_SIZE

RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
#CALIB_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
CALIB_DIR = os.path.join(RAW_DATASET_DIR, 'test_v2')
dir_data = cfg.DATASET_DIR
#b_calib_csv = pd.read_csv(os.path.join(dir_data,"calib.csv"))
b_calib_csv = pd.read_csv(os.path.join(dir_data,"test.csv"))

calib_batch_size = cfg.BATCH_SIZE

name_idx_df = b_calib_csv.set_index('ImageId')
img_ids = np.array( b_calib_csv.index.unique().tolist() )
n_imgs  = img_ids.shape[0]


def calib_input_dbg(iter):  
  x_calib = dus.batch_data_get_images(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None)
  return {"input_1": x_calib}

def calib_input_dbg4(iter):  
  x_calib = dus.batch_data_get_images(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None)
  return {"data": x_calib}


def calib_input1(iter):
  x_calib = dus.batch_data_get_images_all(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None)
  random_indices = np.random.choice(n_imgs//3, size=calib_batch_size, replace=False)
  return {"input_1": x_calib[random_indices]}


def calib_input2(iter): 
  x_calib = dus.batch_data_get_images_all(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None) 
  random_indices = np.random.choice(n_imgs//3, size=calib_batch_size, replace=False)
  return {"input_1": x_calib[random_indices]}

def calib_input3(iter):  
  x_calib = dus.batch_data_get_images_all(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None)
  random_indices = np.random.choice(n_imgs//3, size=calib_batch_size, replace=False)
  return {"input_1": x_calib[random_indices]}

def calib_input4(iter):  
  x_calib = dus.batch_data_get_images_all(b_calib_csv, CALIB_DIR, cfg.BATCH_SIZE, augmentation=None)
  random_indices = np.random.choice(n_imgs//3, size=calib_batch_size, replace=False)
  return {"data": x_calib[random_indices]}

#######################################################

def main():
  calib_input4()


if __name__ == "__main__":
    main()
