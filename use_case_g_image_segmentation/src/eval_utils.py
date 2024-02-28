import cv2, os
import numpy as np
import tensorflow as tf
import sys, time, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Add,Dropout,MaxPooling2D,Input
from tensorflow.keras.layers import Activation,UpSampling2D, Conv2DTranspose

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from config import net_config as cfg

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    Yi[Yi>1] = 1
    
    TP = np.sum(Yi * y_predi)
    FP = np.sum((1-Yi) * y_predi)
    FN = np.sum(Yi * (1-y_predi))
    num = TP
    denum = float(TP + FP + FN)
    if (denum != 0.0):
        IoU = num/denum
    else:
        IoU = 0.0

    #print("TP: ", TP, "FP: ", FP, "FN: ", FN, "IoU", IoU)

    return IoU

def IoU_all(Yi_list,y_predi_list):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    for idx in range (Yi_list.shape[0]):
        Yi = Yi_list[idx]
        y_predi = y_predi_list [idx]

        Yi[Yi>1] = 1
        TP = np.sum(Yi * y_predi)
        FP = np.sum((1-Yi) * y_predi)
        FN = np.sum(Yi * (1-y_predi))

        num = TP
        denum = float(TP + FP + FN)
        if (denum != 0.0):
            IoU = num/denum
        else:
            IoU = 0.0

        IoUs.append(IoU)


    mIoU = np.mean(IoUs)

    return mIoU