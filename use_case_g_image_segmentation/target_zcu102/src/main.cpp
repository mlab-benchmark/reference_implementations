#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

#include <glog/logging.h>

/* header file for Vitis AI unified API */
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <cinttypes>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xadapt.hpp"

#include "unet.h"



int main(int argc, char* argv[]) {

  /*Check the input arguments*/
  if (argc != 6) {

    std::cout << "Usage: unet xmodel_path images_path number_threads number_frames backbone" << std::endl;
    std::cout << "Give the full path for the xmodel and the images" << std::endl;
    std::cout << "Backbone can be resnet50 or mobilenetv2" << std::endl;
    return -1;
  }


  /*Get the input arguments*/
  auto graph = xir::Graph::deserialize(argv[1]);
  std::string imageDirPath = std::string(argv[2]); 
  unsigned int numThreads =  std::stoi(std::string(argv[3]));
  unsigned int numFrames = std::stoi(std::string(argv[4]));
  std::string backbone = std::string(argv[5]);

  /*List with the names of every image*/
  std::vector<std::string> listImageName;

  /*List with runtimes measured for different number of threads*/
  std::vector<std::chrono::milliseconds> runtimeList;

  /*List with frames per second for different number of threads*/
  std::vector<int> fpsList;


  /*Copy the names of the images from the directory to the listImageName*/
  list_images(imageDirPath, listImageName, numFrames);

  /*List with the full path of every image*/
  std::vector<std::string> fullImageDirPath;

  /*Get the full path of every image and add it to the fullimageDirPath*/
  for(auto i = listImageName.begin(); i != listImageName.end(); i++) {

    std::string temp = imageDirPath;

    std::string fullimagePath = temp.append("/").append(*i);
    fullImageDirPath.push_back(fullimagePath);
  }


  auto dpuSubgraphs = get_subgraph(graph.get(), "DPU");
  auto cpuSubgraphs = get_subgraph(graph.get(), "CPU");

  /*Print the subgraph for every processing unit and their input tensors and output tensors*/
  print_data(dpuSubgraphs, cpuSubgraphs);

  /*List with every I/O tensor for every subgraph, every I/0 tensor is 4 dimensional (numFrames, height, width, batch),
    inputTensorList[0][0][1] means the the second 2D tensor of the first input tensor corresponding to the first subgraph*/

  std::array<std::array<std::array<xt::xarray<int>>>> inputTensorList(dpuSubgraphs.size());
  std::array<std::array<std::array<xt::xarray<int>>>> outputTensorList(dpuSubgraphs.size());

  prepare_data_dpu(dpuSubgraphs, numFrames, inputTensorList, outputTensorList);





  /*Run the UNet for different number of threads*/
  for(unsigned int i = 1; i <= numThreads; i++) {

    std::cout << "Processing for " << i << " threads" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    run_unet(backbone, numThreads, fullImageDirPath, dpuSubgraphs, inputTensorList, outputTensorList);

    auto endTime = std::chrono::high_resolution_clock::now();

    /*Get the runtime of the UNet*/
    std::chrono::milliseconds runtime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    runtimeList.push_back(runtime);

    /*Get the frames per second*/
    int fps = numFrames / ((double)runtime.count() / 1000);
    fpsList.push_back(fps);
  }




  /*Write the metrics to a csv file*/
  write_csv(backbone, fpsList, runtimeList, numThreads);
  
  return 0;
}
