# from ctypes import *
# from typing import List
# import cv2
# import math
# import threading
# import sys
# import queue
# from hashlib import md5
import numpy as np
import xir
import vart
import vitis_ai_library
import os
import time
import argparse


def app(model_name):

    time.sleep(2)

    graph = xir.Graph.deserialize(model_name)
    runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)

    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    X_test = np.load('X_test.npy').astype('float32')

    pred_data = X_test
    output_length = pred_data.shape[0]
    output_data = np.zeros([output_length, 1, 1, 1], dtype='float32')
    count = 0
    time1 = time.time()
    processes = set()
    while count < output_length:
        # time11=time.time()
        p = runner.execute_async(pred_data[count], output_data[count])
        processes.add(p)
        count = count + 1

    # Check that all processes completed
    for p in processes:
        runner.wait(p)

    time2 = time.time()
    total_time = time2 - time1
    time_out = f'Start time: {time1:.4f} - End time: {time2:.4f} - Total time: {total_time:.4f}'
    print(time_out)
    print("count value", count)
    np.save('./y_pred.npy', output_data)

    # write time_out to file in output
    if not os.path.exists('output'):
        os.makedirs('output')
    processor_times_path = 'output/processor_times.npy'
    if os.path.exists(processor_times_path):
        processor_times = np.load(processor_times_path)
        processor_times = np.append(processor_times, np.array([[time1, time2]]), axis=0)
        np.save(processor_times_path, processor_times)
    else:
        processor_times = np.array([[time1, time2]])
        np.save(processor_times_path, processor_times)

    # for test data
    pred_data = X_test
    print("Test data prediction")
    output_length = len(pred_data[:])
    # output_length = 1
    print("ouput length =", output_length)

    return



def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', type=str, default='./nbeats_forecast.xmodel',
                    help='Path of xmodel')

    args = ap.parse_args()
    print("\n")
    print('Command line options:')
    # print (' --threads    : ', args.threads)
    print(' --model      : ', args.model)
    print("\n")

    # app(args.images_dir,args.threads,args.model)
    time.sleep(3)
    app(args.model)


if __name__ == '__main__':
    main()
