from ctypes import *
from typing import List
import cv2
import numpy as np
import os
import pathlib
import threading
import time
import sys
import xir
import argparse
import math
import vart
import csv


#Get the directory of the xmodel
xmodelDir = os.getcwd() + "/model_dir/unet4_resnet50.xmodel"

#Directory of test images
imageDir = os.getcwd() + "/test_v2/"
imageList = os.listdir(imageDir)


#Sigmoid layer running in CPU
def cpu_sigmoid_calc(data, size):

    output = []

    for i in range(size):
        output.append(1.0 / (1.0 + np.exp(-data[i])))

    return output


#Padding layer running in CPU, 2D padding with zeros
def cpu_zero_pad(data, size):

    output = []

    for i in range(size):

        output.append(np.pad(data[i], ((1, 1), (1, 1), (0, 0)), 'constant')) 

    return output


#Read an image, convert to RGB and scale it
def preprocess_image(imagePath, width = 224, height = 224):

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (width, height))

    #OpenCV uses BGR, convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


#Get the subgraphs, that should run in the CPU or in the DPU
def get_child_subgraph(graph: "Graph", processUnit) -> List["Subgraph"]:

    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph()

    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return []

    child_subgraphs = root_subgraph.toposort_child_subgraph()

    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [cs for cs in child_subgraphs if (cs.has_attr("device") and 
            cs.get_attr("device").upper() == processUnit)]



#Get the whole graph from the xmodel and get subgraphs of DPU and CPU
graph = xir.Graph.deserialize(xmodelDir)
dpuSubgraphs = get_child_subgraph(graph, "DPU")
cpuSubgraphs = get_child_subgraph(graph, "CPU")


def print_data(dpuSubgraphs, cpuSubgraphs):

    print("DPU Subgraphs: ", len(dpuSubgraphs))
    print("CPU Subgraphs: ", len(cpuSubgraphs))

    print("DPU subgraphs:")
    for id, item in enumerate (dpuSubgraphs):
        print(f'Subgraph {id}: {item.get_name()}')
        for jtem in item.get_input_tensors():
            print(f'Input tensors: {tuple(jtem.dims)}')
        for jtem in item.get_output_tensors():
            print(f'Output tensors: {tuple(jtem.dims)}')


    print("CPU subgraphs:")

    for id, item in enumerate (cpuSubgraphs):
        print(f'Subgraph {id}: {item.get_name()}')
        for jtem in item.get_input_tensors():
            print(f'Input tensors: {tuple(jtem.dims)}')
        for jtem in item.get_output_tensors():
            print(f'Output tensors: {tuple(jtem.dims)}')



#Get the name attribute of the given object
def get_name(x):

    return x.name



def process_dpu_subgraph(runner: "Runner", image, numFrames, start, end, subgraphID):

    #Get tensors
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    inputDim = tuple(inputTensors[0].dims)
    outputDim = tuple(outputTensors[0].dims)


    count = start

    while(count < end):

        runSize = inputDim[0]

        #Batch I/O
        inputData = [np.ones(tuple(inputTensors[k].dims), dtype = np.float32, order = "C")
                     for k in range(len(inputTensors))]

        outputData = [np.empty(tuple(outputTensors[k].dims), dtype = np.float32, order = "C")
                      for k in range(len(outputTensors))]


        #Init input image to input buffer
        for k in range(len(image)):
            for j in range(runSize):
                imageRun = inputData[k]
                imageRun[j, ...] = image[k][(count + j) % numFrames].reshape(tuple(inputTensors[k].dims)[1:])

        #Run with batch
        jobID = runner.execute_async(inputData, outputData)
        runner.wait(jobID)

        for k in range(len(outputTensors)):
            for j in range(runSize):
                outputImgDpu[subgraphID][k][(count + j) % numFrames] = outputData[k][j]

        count += runSize


#Prepare I/O tensors for dpu subgraphs
def prepare_data_dpu(count):

    #prepare a list of input tensors for DPU subgraphs
    inputTensors = []
    outputTensors = []
    tmpList = []

    for item in dpuSubgraphs:

        tmp = item.get_input_tensors()
        tmpList = sorted([i for i in tmp], key = get_name)
        inputTensors.append(tmpList)
        
        tmp = item.get_output_tensors()
        tmpList = sorted([i for i in tmp], key = get_name)
        outputTensors.append(tmpList)


    global inputImgDpu
    inputImgDpu = []

    global outputImgDpu
    outputImgDpu = []

    # allocating input and output for DPU subgraphs
    print(f'Input tensors:')
    for item in inputTensors:

        inputImgDpu.append([])

        for jtem in item:
            
            dataSize = jtem.dims.copy()
            dataSize[0] = count
            dataSize = tuple(dataSize)
            inputImgDpu[-1].append(np.zeros(dataSize, dtype = np.float32))
            print(dataSize)    


    print(f'Output tensors:')
    for item in outputTensors:

        outputImgDpu.append([])

        for jtem in item:
            
            dataSize = jtem.dims.copy()
            dataSize[0] = count
            dataSize = tuple(dataSize)
            outputImgDpu[-1].append(np.zeros(dataSize, dtype = np.float32))
            print(dataSize)    


def run_app(numThreads, numFrames):

    startList = []
    endList = []

    start = 0

    for i in range(numThreads):

        if i == numThreads - 1:
            end = numFrames
        else:
            end = start + (numFrames // numThreads)

        startList.append(start)
        endList.append(end)

        start = end



    startTime = time.time_ns() // 1000000

    #######################################################
    # CPU subgraph 0 
    #######################################################

    width = inputImgDpu[0][0].shape[1]
    height = inputImgDpu[0][0].shape[2]


    #Preprocess the images 
    for i in range(numFrames):

        path = os.path.join(imageDir, imageList[i])
        inputImgDpu[0][0][i] = preprocess_image(path, width, height)
    
    #######################################################
    # DPU subgraph 0
    #######################################################

    #Take the preprocessed images as input to the subgraph 0 of the DPU
    inputs = [inputImgDpu[0][0]]


    threadAll = []
    dpuRunner = []

    subgraphID = 0

    #Create a DPU runner for every thread
    for i in range(int(numThreads)):
        dpuRunner.append(
            vart.Runner.create_runner(dpuSubgraphs[subgraphID], "run"))

    """run with batch """
    for i in range(int(numThreads)):

        t1 = threading.Thread(target = process_dpu_subgraph, 
            args = (dpuRunner[i], inputs, numFrames, startList[i], endList[i], subgraphID))

        threadAll.append(t1)

    for x in threadAll:
        x.start()

    for x in threadAll:
        x.join()

    del dpuRunner

    #######################################################
    # CPU subgraph 1
    #######################################################

    inputImgDpu[1][0] = cpu_zero_pad(outputImgDpu[0][0], numFrames)

    
    #######################################################
    # DPU subgraph 1
    #######################################################

    inputs = [outputImgDpu[0][0], inputImgDpu[1][0]]

    threadAll = []
    dpuRunner = []

    subgraphID = 1

    for i in range(numThreads):

        dpuRunner.append(
            vart.Runner.create_runner(dpuSubgraphs[subgraphID], "run"))

    for i in range(int(numThreads)):

        t1 = threading.Thread(target = process_dpu_subgraph, 
            args = (dpuRunner[i], inputs, numFrames, startList[i], endList[i], subgraphID))

        threadAll.append(t1)

    for x in threadAll:
        x.start()

    for x in threadAll:
        x.join()

    del dpuRunner

    #######################################################
    # CPU subgraph 2
    #######################################################

    prediction = cpu_sigmoid_calc(outputImgDpu[1][0], numFrames)
    
    #print("Prediction is : " + str(prediction))
    print("Prediction shape is : " + str(np.shape(prediction)))
    
    with open('res.npy', 'wb') as f:
    	np.save(f, np.array(prediction))

    endTime = time.time_ns() // 1000000
    totalTime = endTime - startTime

    fps = int(numFrames / (totalTime / 1000))

    metrics = [numThreads, fps, totalTime]

    
    return metrics



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument('-t', '--numThreads',   type = int, default = 1,        help = 'Number of numThreads. Default is 1')
    ap.add_argument('-f', '--numFrames',     type = int, default = 100, help = 'Number of frames to run')

    args = ap.parse_args()  
  
    print('Command line options:')
    print(' --numThreads   : ', args.numThreads)
    print(' --numFrames    : ', args.numFrames)

    #Metrics to read after the processing of the CNN
    metrics = []
    allMetrics = []

    header = ['Number of threads', 'fps', 'runtime']

    print_data(dpuSubgraphs, cpuSubgraphs)

    prepare_data_dpu(args.numFrames)

    for i in range(1, args.numThreads + 1):

        metrics = run_app(i, args.numFrames)
        allMetrics.append(metrics)

    #Write the date into a csv file
    with open('metrics_resnet50.csv', 'w', encoding = 'UTF8') as f:
        writer = csv.writer(f)

        #Write the header
        writer.writerow(header)

        #Write the metrics for every thread number
        for i in range(args.numThreads):
            writer.writerow(allMetrics[i])


if __name__ == '__main__':
    main()
