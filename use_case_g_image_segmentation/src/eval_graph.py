#usr/bin/env python
# -*- coding: utf-8 -*-


##################################################################
# Evaluation of frozen/quantized graph
#################################################################

import os
import sys
import glob
import argparse
import shutil
import tensorflow as tf
import numpy as np
import cv2
import gc # memory garbage collector #DB
import pandas as pd
import data_utils as dus

import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm

# reduce TensorFlow messages in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tensorflow.contrib.decent_q

from tensorflow.python.platform import gfile
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend               import set_session
from tensorflow import keras as K

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
set_session(tf.compat.v1.Session(config=config))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#config.gpu_options.visible_device_list = "1"
#K.backend.set_session(tf.compat.v1.Session(config=config))

from config import net_config as cfg
from config import fcn8_cnn as cnn
import eval_utils as eus

def get_script_directory():
    path = os.getcwd()
    return path

DATAS_DIR     = cfg.DATASET_DIR
IMG_TEST_DIR  = cfg.dir_test_img
SEG_TEST_DIR  = cfg.dir_test_seg

HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
N_CLASSES = cfg.NUM_CLASSES
BATCH_SIZE = cfg.BATCH_SIZE

RAW_DATASET_DIR = cfg.RAW_DATASET_DIR
TEST_DIR = os.path.join(RAW_DATASET_DIR, 'train_v2')
dir_data = cfg.DATASET_DIR
b_test_csv = pd.read_csv(os.path.join(dir_data,"calib.csv"))

def graph_eval(input_graph_def, input_node, output_node, model_type):
    #Reading images and segmentation labels
    #x_test, y_test, img_file, seg_file = cnn.get_images_and_labels(IMG_TEST_DIR, SEG_TEST_DIR, cfg.NUM_CLASSES, cfg.WIDTH, cfg.HEIGHT)

    x_test, y_test = dus.batch_data_get(b_test_csv, TEST_DIR, cfg.BATCH_SIZE, augmentation=None)
    #x_test, y_test = dus.batch_data_get_all(b_test_csv, TEST_DIR, cfg.BATCH_SIZE, augmentation=None)

    # load graph
    tf.import_graph_def(input_graph_def,name = '')

    # Get input & output tensors
    x = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node+':0')
    y = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node+':0')

    # Create the Computational graph
    y_pred=np.zeros(y_test.shape)
    with tf.compat.v1.Session() as sess:
        #sess.run(tf.compat.v1.initializers.global_variables())
        #feed_dict={x: x_test} #, labels: y_test}
        #y_pred = sess.run(y, feed_dict)
        for idx in tqdm( range(0, x_test.shape[0], cfg.BATCH_SIZE)):
            sess.run(tf.compat.v1.initializers.global_variables())
            feed_dict={x: x_test[idx*cfg.BATCH_SIZE:(idx+1)*cfg.BATCH_SIZE]} #, labels: y_test}
            y_pred[idx*cfg.BATCH_SIZE:(idx+1)*cfg.BATCH_SIZE] = sess.run(y, feed_dict)

    # Calculate intersection over union for each segmentation class
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test, axis=3)
    #print(y_testi.shape,y_predi.shape)

    #print ("x_test.shape: ", x_test.shape)
    #print ("y_test.shape: ", y_test.shape)
    #print ("y_pred.shape: ", y_pred.shape)
    fig, ax = plt.subplots(1,3)
    im0 = ax[0].imshow(x_test[0])
    #plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(y_test[0])
    #plt.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(y_pred[0])
    #plt.colorbar(im2, ax=ax[2])
    IoU = eus.IoU(y_test[0],y_pred[0])
    plt.suptitle('Prediction, IoU: ' + str(IoU))
    plt.savefig("../rpt/unet_model_progress_12_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
    plt.show()

    n_samples = min (25, cfg.BATCH_SIZE)
    fig, axs = plt.subplots(ncols=3, nrows=n_samples, figsize=(5, 25), sharex=True, sharey=True)
    #for idx,jdx in enumerate (random.sample(range(0,y_pred.shape[0]), n_samples)):
    for idx,jdx in enumerate (range(0,n_samples)):
        IoU = eus.IoU(y_test[jdx],y_pred[jdx])
        axs[idx,0].imshow(x_test[jdx])
        axs[idx,1].imshow(y_test[jdx])
        axs[idx,2].imshow(y_pred[jdx])
        #axs[idx,2].set_title('IoU: ' + str(IoU))

    axs[0,0].set_title('Input')
    axs[0,1].set_title('Mask')
    axs[0,2].set_title('Prediction')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("../rpt/unet_model_progress_13_m_" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")
    plt.show()

    mIoU = eus.IoU_all (y_test,y_pred)
    print ("IoU for original model for the test dataset: ", mIoU)

    print ('FINISHED!')
    #return x_test, y_testi, y_predi, img_file, seg_file


def main(unused_argv):

    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.io.gfile.GFile(FLAGS.graph, "rb").read())
    #x_test,y_testi,y_predi,img_file,seg_file = graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)
    graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node, FLAGS.model)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str,
                        default="../freeze/frozen_graph.pb",
                        help="graph file (.pb) to be evaluated.")
    parser.add_argument("--input_node", type=str,
                        default="input_1",
                        help="input node.")
    parser.add_argument("--output_node", type=str,
                        default="activation_1/truediv",
                        help="output node.")
    parser.add_argument("--class_num", type=int,
                        default=cfg.NUM_CLASSES,
                        help="number of classes.")
    parser.add_argument("--gpu", type=str,
                        default="0",
                        help="gpu device id.")
    parser.add_argument("--model", type=int,
                        default="0",
                        help="model id.")
    FLAGS, unparsed = parser.parse_known_args()

    #tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main(unparsed)
