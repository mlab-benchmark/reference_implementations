#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import sys, warnings
from datetime import datetime #DB

from config import net_config as cfg
from config import unet as unet #DB
import data_utils as dus #DB

import pandas as pd
warnings.filterwarnings("ignore")

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
## Import usual libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras                       import backend
from tensorflow.keras.utils                 import plot_model #DB
from tensorflow.keras.models                import *
from tensorflow.keras.layers                import *
from tensorflow.keras.optimizers            import RMSprop, SGD
from tqdm.auto import tqdm

######################################################################
from segmentation_models.losses    import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics   import iou_score, f2_score

import segmentation_models as sm

tf.compat.v1.disable_eager_execution()

sm.set_framework('tf.keras')

sm.framework()

######################################################################
IMG_HW    = cfg.IMG_HW
BACKBONE  = cfg.BACKBONE

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES

BATCH_SIZE = cfg.BATCH_SIZE
STEPS_PER_EPOCH = cfg.STEPS_PER_EPOCH
VAL_STEPS = cfg.VAL_STEPS
EPOCHS = cfg.EPOCHS
######################################################################

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.85
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0"
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m",  "--model", default=False, help="models: 1, 2, 3 or 4")
args = vars(ap.parse_args())
model_type= int(args["model"])
print("UNET MODEL = ", model_type)

######################################################################
# directories
######################################################################
RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
TRAIN_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
dir_data = cfg.DATASET_DIR

b_train_csv = pd.read_csv(os.path.join(dir_data,"train.csv"))
b_valid_csv = pd.read_csv(os.path.join(dir_data,"valid.csv"))
######################################################################
# model a
######################################################################

if (model_type==1):
        model = unet.UNET_v1(N_CLASSES, HEIGHT, WIDTH)
elif (model_type==2):
        model = unet.UNET_v2(N_CLASSES, HEIGHT, WIDTH)
elif (model_type==3):
        model = unet.UNET_v3(N_CLASSES, HEIGHT, WIDTH)
else:
        model = unet.UNET_v4(N_CLASSES, HEIGHT, WIDTH)
model.summary()


######################################################################
# it is a nasty way to deal with this, but works for now
if (model_type==1 or model_type==2 or model_type==3):
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler
else:
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler
######################################################################

# plot the CNN model
plot_model(model, to_file="../rpt/unet_model_m_" +str(model_type) + "_"  + str(WIDTH) + "x" + str(HEIGHT) + ".png", show_shapes=True)

optimizer = 'Adam'
#optimizer = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

loss = bce_dice_loss
#loss='categorical_crossentropy'

metrics = [iou_score]
#metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

file_path_best_checkpoint = '../keras_model/unet/best_model_'+str(model_type)+'.hdf5'
checkpoint = ModelCheckpoint(
    filepath=file_path_best_checkpoint, 
    monitor='val_iou_score', mode='max', 
    save_best_only=True, save_weights_only=False, 
    verbose=1
)

reduce_lr  = ReduceLROnPlateau(
    monitor='val_loss', mode='min', 
    factor=0.3, patience=3, min_lr=0.000001, 
    verbose=1
)
callbacks_list=[ checkpoint, reduce_lr ]
#callbacks_list = []


startTime1 = datetime.now() #DB
history = model.fit_generator(
    generator        = dus.batch_data_gen(b_train_csv, TRAIN_DIR, BATCH_SIZE, augmentation=None), 
    validation_data  = dus.batch_data_gen(b_valid_csv, TRAIN_DIR, BATCH_SIZE), 
    validation_steps = VAL_STEPS,
    steps_per_epoch  = STEPS_PER_EPOCH,
    epochs           = EPOCHS,
    verbose = 1,
    callbacks = callbacks_list
)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

los  = model.history.history['loss']
vlos = model.history.history['val_loss']
iou  = model.history.history['iou_score']
viou = model.history.history['val_iou_score']

epochs = np.arange(1, len(los)+1)
plt.plot(epochs, los,  label='Training loss')
plt.plot(epochs, vlos, label='Validation loss')
plt.legend()
plt.savefig("../rpt/unet_model_progress_07_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
plt.show()

epochs = np.arange(1, len(los)+1)
plt.plot(epochs, iou,  label='IoU')
plt.plot(epochs, viou, label='Validation IoU')
plt.legend()
plt.savefig("../rpt/unet_model_progress_08_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
plt.show()

# fine_tunning
if (cfg.ENABLE_FINE_TUNNING):
    model.load_weights(file_path_best_checkpoint)
    lrs = LearningRateScheduler(lambda epoch: 0.001 * 0.3 ** (epoch // 2))   
    callbacks_list_fine_tunning=[ checkpoint, lrs ]

    history = model.fit_generator(
        generator        = dus.batch_data_gen(b_train_csv, TRAIN_DIR, BATCH_SIZE, augmentation=dus.augmentor), 
        validation_data  = dus.batch_data_gen(b_valid_csv, TRAIN_DIR, BATCH_SIZE), 
        validation_steps = VAL_STEPS,
        steps_per_epoch  = STEPS_PER_EPOCH,
        epochs           = EPOCHS//2,
        verbose = 1,
        callbacks= callbacks_list_fine_tunning,
    )

    los  = model.history.history['loss']
    vlos = model.history.history['val_loss']
    iou  = model.history.history['iou_score']
    viou = model.history.history['val_iou_score']

    epochs = np.arange(1, len(los)+1)
    plt.plot(epochs, los,  label='Training loss')
    plt.plot(epochs, vlos, label='Validation loss')
    plt.legend()
    plt.savefig("../rpt/unet_model_progress_09_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
    plt.show()

    epochs = np.arange(1, len(los)+1)
    plt.plot(epochs, iou,  label='IoU')
    plt.plot(epochs, viou, label='Validation IoU')
    plt.legend()
    plt.savefig("../rpt/unet_model_progress_10_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
    plt.show()

for key in ["loss", "val_loss"]:
    plt.plot(history.history[key],label=key)
plt.legend()
plt.savefig("../rpt/unet_model_progress_11_m_" + str(model_type) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")

model.save("../keras_model/unet/ep" + str(EPOCHS) + "_trained_unet_model" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5")


print("\nEnd of UNET training\n")
