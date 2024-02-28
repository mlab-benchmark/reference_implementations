#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
## Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from random import random
import sys, warnings
import pandas as pd
import csv

from config import net_config as cfg
import data_utils as dus

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras                       import backend
from tensorflow.keras.models                import *
from tensorflow.keras.layers                import *

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

######################################################################
import os, sys

import pandas as pd
import numpy  as np

import seaborn as sns
import matplotlib.pyplot as plt

######################################################################

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.85
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0"
#set_session(tf.compat.v1.Session(config=config))

print("\n")
print("python {}".format(sys.version))
print("keras version {}".format(tf.keras.__version__)); 
print("tensorflow version {}".format(tf.__version__))
print(backend.image_data_format())
print("\n")

######################################################################
# prepare data and masks
######################################################################
RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
TRAIN_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
TEST_DIR  = os.path.join(RAW_DATASET_DIR, 'test_v2')
SEG_DIR = os.path.join( RAW_DATASET_DIR, 'train_ship_segmentations_v2.csv')
dir_data = cfg.DATASET_DIR

train_csv  = pd.read_csv( SEG_DIR )
train_csv['withShip'] = train_csv['EncodedPixels'].isnull()
train_csv['npixel'] = dus.count_pix_inpool( train_csv['EncodedPixels'] )

DROP_NO_SHIP_FRACTION = 0.8

plt.figure()
sns.kdeplot(train_csv['npixel'])
plt.xscale('log')
plt.legend().set_visible(False)
plt.xlabel('# of pixel is ship')
plt.ylabel('Density')
plt.savefig("../rpt/unet_model_progress_01_" + ".png")
plt.show()

plt.figure()
plt.pie(
    x       = (train_csv['withShip'].value_counts()/train_csv.shape[0]).values, 
    labels  = (train_csv['withShip'].value_counts()/train_csv.shape[0]).index,
    autopct = '%3.1f %%',
    shadow  = True, 
    labeldistance = 1.1, 
    startangle  = 90,
    pctdistance = 0.6
)
plt.title('Image with Ship(s) or Not')
plt.savefig("../rpt/unet_model_progress_02_" + ".png")
plt.show()

plt.figure()
figdf = train_csv.dropna().groupby('ImageId').count()['withShip'].value_counts()
plt.bar(figdf.index, figdf.values)
plt.xlabel('# of Ship(s) in image', fontsize=14)
plt.ylabel('# of Images', fontsize=14)
plt.title('Raw Dataset with Ship', fontsize=18)
plt.savefig("../rpt/unet_model_progress_03_" + ".png")
plt.show()

balanced_train_csv = (
    train_csv
    .set_index('ImageId')
    .drop(
        train_csv.loc[
            train_csv.isna().any(axis=1),
            'ImageId'
        ].sample( frac = DROP_NO_SHIP_FRACTION )
    )
    .drop(
        train_csv.query('npixel<32')['ImageId']
    )
    .reset_index()
)

plt.figure()
plt.pie(
    x       = (balanced_train_csv['withShip'].value_counts()/balanced_train_csv.shape[0]).values, 
    labels  = (balanced_train_csv['withShip'].value_counts()/balanced_train_csv.shape[0]).index,
    autopct = '%3.1f %%',
    shadow  = True, 
    labeldistance = 1.1, 
    startangle  = 90,
    pctdistance = 0.6
);
plt.title('Image with Ship(s) or Not');
plt.savefig("../rpt/unet_model_progress_04_" + ".png")
plt.show()

plt.figure()
figdf = balanced_train_csv.fillna(0).groupby('ImageId').sum()['withShip'].value_counts()
plt.bar(figdf.index, figdf.values)
plt.xlabel('# of Ship(s) in image', fontsize=14)
plt.ylabel('# of Images', fontsize=14)
plt.title('Balanced Dataset w/o Ship', fontsize=18)
plt.savefig("../rpt/unet_model_progress_05_" + ".png")
plt.show()

fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(20, 20), sharex=True, sharey=True)

for i, img_id in enumerate( np.random.choice(train_csv.dropna()['ImageId'].unique(), 5) ):
#     print (img_id)
    img_df = train_csv.set_index('ImageId').loc[[img_id]]
    x, y = dus.load_paired_data(img_df, TRAIN_DIR)

    axs[i,0].set_ylabel(img_id)
    axs[i,0].imshow(x)
    axs[i,1].imshow(y[:,:,0])

axs[0,0].set_title('Input')
axs[0,1].set_title('Mask')

plt.xticks([])
plt.yticks([])
plt.savefig("../rpt/unet_model_progress_06_" + ".png")
plt.show()

b_train_csv, b_valid_csv = train_test_split(balanced_train_csv['ImageId'], test_size = 0.2)
b_train_csv = balanced_train_csv.set_index('ImageId').loc[b_train_csv].reset_index()
b_valid_csv = balanced_train_csv.set_index('ImageId').loc[b_valid_csv].reset_index()
b_calib_csv = b_train_csv.sample(frac=0.05).reset_index(drop=True)

test_csv = pd.DataFrame (columns = train_csv.columns)
test_csv ['ImageId'] = os.listdir(TEST_DIR)

b_train_csv.to_csv(os.path.join(dir_data,"train.csv"))
b_valid_csv.to_csv(os.path.join(dir_data,"valid.csv"))
b_calib_csv.to_csv(os.path.join(dir_data,"calib.csv"))
test_csv.to_csv(os.path.join(dir_data,"test.csv"))

with open(os.path.join(dir_data,"train.csv"), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)


