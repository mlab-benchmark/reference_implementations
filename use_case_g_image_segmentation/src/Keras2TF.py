

'''
/**

* © Copyright (C) 2016-2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
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


# USAGE
# python Keras2TFy -w weights_file.hdf52

import os
import sys
import shutil

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, load_model

from config import net_config as cfg

from segmentation_models.losses    import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics   import iou_score, f2_score
import tensorflow.keras.losses
######################################################################

##################################################################################

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m",  "--model", help="CNN Models: fcn8, fcn8ups, unet1, unet2, unet3, unet4")

args = vars(ap.parse_args())

model_name = args["model"]

##############################################
# Set up directories
##############################################

KERAS_MODEL_DIR = cfg.KERAS_MODEL_DIR #DB

WEIGHTS_DIR = KERAS_MODEL_DIR

CHKPT_MODEL_DIR = cfg.CHKPT_MODEL_DIR

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES

BATCH_SIZE = cfg.BATCH_SIZE
EPOCHS = cfg.EPOCHS


# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)


# load weights & architecture into new model
if model_name=="fcn8ups" :
        weights= "fcn8ups/ep2_trained_fcn8ups_224x224.hdf5"
elif model_name=="fcn8" :
        weights= "fcn8/ep2_trained_fcn8_224x224.hdf5"
elif model_name=="unet1" :
        weights= "unet/ep" + str(EPOCHS) + "_trained_unet_model" + str(1) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"
elif model_name=="unet2" :
        weights= "unet/ep" + str(EPOCHS) + "_trained_unet_model" + str(2) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"
elif model_name=="unet3" :
        weights= "unet/ep" + str(EPOCHS) + "_trained_unet_model" + str(3) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"
else: # elif model_name=="unet4" :
        weights= "unet/" + "best_model_" + str(4) + ".hdf5"

print("model name = ", model_name)
filename = os.path.join(WEIGHTS_DIR,weights)

assert os.path.isdir(WEIGHTS_DIR)
assert os.path.isfile(filename)


model = load_model(filename, custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_score':iou_score},
                   compile = False) 

##print the CNN structure
#model.summary()

# make list of output node names
output_names=[out.op.name for out in model.outputs]

# set up tensorflow saver object
saver = tf.compat.v1.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = tf.compat.v1.keras.backend.get_session()

# Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)

# write out tensorflow checkpoint & meta graph
if model_name=="fcn8ups" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fcn8ups/float_model.ckpt"))
elif model_name=="fcn8" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fcn8/float_model.ckpt"))
elif model_name=="unet1" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet1/float_model.ckpt"))
elif model_name=="unet2" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet2/float_model.ckpt"))
elif model_name=="unet3" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet3/float_model.ckpt"))
else: # elif model_name=="unet4" :
    	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet4/float_model.ckpt"))

print ("\nFINISHED CREATING TF FILES\n")
