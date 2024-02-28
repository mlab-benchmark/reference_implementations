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





/*Print the subgraph for every processing unit and their input tensors and output tensors*/
void print_data(std::vector<const xir::Subgraph*> dpuSubgraphs, std::vector<const xir::Subgraph*> cpuSubgraphs) {


  std::cout << "There are" << " " << dpuSubgraphs.size() << " DPU subgraphs:" << std::endl;

  unsigned int subgraphID = 0;

  for(auto subgraph = dpuSubgraphs.begin(); subgraph != dpuSubgraphs.end(); subgraph++) {

    std::cout << "Subgraph " << subgraphID << ": " << (*subgraph) -> get_name() << std::endl;

    /*For every subgraph print their input tensors and output tensors*/
    std::cout << "Input tensors:" << std::endl;

    std::set<const xir::Tensor*> inputTensors = (*subgraph) -> get_input_tensors();

    for(auto j: inputTensors) {

      const std::vector<int> inputDim = j -> get_shape();

      std::cout << "(";
      for(auto i = inputDim.begin(); i != inputDim.end(); i++) {

        if(i == inputDim.end() - 1)
          std::cout << *i << ")" << std::endl;

        else
          std::cout << *i << ", ";
      }
    }


    std::cout << "Output tensors:" << std::endl;

    std::set<const xir::Tensor*> outputTensors = (*subgraph) -> get_output_tensors();

    for(auto j: outputTensors) {

      const std::vector<int> outputDim = j -> get_shape();
      
      std::cout << "(";
      for(auto i = outputDim.begin(); i != outputDim.end(); i++) {

        if(i == outputDim.end() - 1)
          std::cout << *i << ")" << std::endl;

        else
          std::cout << *i << ", ";
      }
    }

    std::cout << std::endl;

    subgraphID++;

  }

  std::cout << std::endl;

  std::cout << "There are" << " " << cpuSubgraphs.size() << " CPU subgraphs:" << std::endl;
  for(unsigned int i = 0; i < cpuSubgraphs.size(); i++)
    std::cout << cpuSubgraphs[i] -> get_name() << std::endl;

}


/*Implementation of 2D_Padding layer, that should run in the CPU*/
void cpu_zero_pad(std::vector<xt::xarray<int>> &inputImages, std::vector<xt::xarray<int>> &outputImages) {

  outputImages.clear();

  for(auto i = inputImages.begin(); i != inputImages.end(); i++)
    outputImages.push_back(xt::pad(*i, {{0, 0}, {1, 1}, {1, 1}}));
}



/*Implementation of the sigmoid layer, that should run in the CPU*/
std::vector<xt::xarray<int>> sigmoid_cpu_calc(std::vector<xt::xarray<int>> images) {

  std::vector<xt::xarray<int>> outputImages;

  for(auto i = images.begin(); i != images.end(); i++)
    outputImages.push_back(*i / (1 + abs(*i)));

  return outputImages;
}


/*Get batchSize images from the image path and pass their names to the imageList*/
void list_images(std::string const& imageDirPath, std::vector<std::string>& imageList, unsigned int batchSize) {

  imageList.clear();
  struct dirent* entry;

  /*Check if imageDirPath is a valid directory path */
  struct stat s;

  lstat(imageDirPath.c_str(), &s);

  if (!S_ISDIR(s.st_mode)) {

    fprintf(stderr, "Error: %s is not a valid directory!\n", imageDirPath.c_str());
    exit(1);
  }


  DIR* dir = opendir(imageDirPath.c_str());

  if(dir == nullptr) {

    fprintf(stderr, "Error: Open %s imageDirPath failed.\n", imageDirPath.c_str());
    exit(1);
  }

  unsigned int i = 0;


  /*Read the images from the directory accordin to the batchSize*/
  while((entry = readdir(dir)) != nullptr && i <= batchSize + 1) {

    if (entry -> d_type == DT_REG || entry -> d_type == DT_UNKNOWN) {

      std::string name = entry -> d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);

      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {

        imageList.push_back(name);
      }
    }

    i++;
  }


  closedir(dir);
}


/*Read the dimensions of a xir::Tensor and return them as a vector*/
std::vector<int> get_tensor_dimensions(const xir::Tensor* tensor) {

  auto dim = tensor -> get_shape();
  std::vector<int> dimList;

  for(auto i = dim.begin() + 1; i != dim.end(); i++)
    dimList.push_back(*i);

  return dimList;
}

/*Read the dimensions of a xir::Tensor and return them as a vector*/
std::size_t get_tensor_size(std::vector<int> dimList) {

  std::size_t tensorSize = 1;

  for(auto i = dimList.begin(); i != dimList.end(); i++)
    tensorSize *= *i;

  return tensorSize;
}



/*Resize the image to the input sizes and convert every pixel to 8 Bits unsigned integers and 
  pass them to xtensor*/
void preprocess_image(std::vector<std::string> fullImageDirPath, std::vector<xt::xarray<int>> &outputImages) {

  outputImages.clear();

  std::vector<int> shapes = {CHANNEL, HEIGHT, WIDTH};
  std::size_t size = HEIGHT * WIDTH * CHANNEL;

  /*Create and empty image with the dimensions height, width and every pixel to 8 Bits unsigned integer*/
  cv::Mat outputImage(HEIGHT, WIDTH, CV_8UC3);

  /*Get the full path of every image and add it to the fullimageDirPath*/
  for(auto i = fullImageDirPath.begin(); i != fullImageDirPath.end(); i++) {

    cv::Mat inputImage = cv::imread(*i);

    /*Copy the resized inputImage to outputImage*/
    cv::resize(inputImage, outputImage, cv::Size(HEIGHT, WIDTH));

    xt::xarray<int> image = xt::adapt((int*)outputImage.data, size, xt::no_ownership(), shapes);

    outputImages.push_back(image);
  }
}



void prepare_data_dpu(std::vector<const xir::Subgraph*> dpuSubgraphs, unsigned int numFrames,
                      std::vector<std::vector<std::vector<xt::xarray<int>>>> &inputTensorList,
                      std::vector<std::vector<std::vector<xt::xarray<int>>>> &outputTensorList) {

  /*Clear the I/O tensors of every subgraphs, as they are initialized with zeros*/
  for(unsigned int i = 0; i < dpuSubgraphs.size(); i++) {

    inputTensorList[i].clear();
    outputTensorList[i].clear();
  }

  /*Iterate through all subgraphs that should run in the DPU*/
  for(unsigned int i = 0; i < dpuSubgraphs.size(); i++) {

    auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[i], "run");

    /*Get the input tensors of the current subgraph*/
    std::vector<const xir::Tensor*> inputTensors = dpuRunner -> get_input_tensors();


    for(auto j: inputTensors) {

      std::vector<int> inputDimList;

      inputDimList.push_back(numFrames);

      const std::vector<int> inputDim = j -> get_shape();

      /*Get the height, width and batch size of the input tensor,
        the first value is the number of frames, which we already passed*/
      for(auto dim = inputDim.begin() + 1; dim != inputDim.end(); dim++)
        inputDimList.push_back(*dim);

      /*List with all the input tensors of the current subgraph*/
      std::vector<xt::xarray<int>> inputTensor(inputDimList[0], xt::zeros<int>({inputDimList[3], inputDimList[1], 
                                                                                inputDimList[2]}));
      inputTensorList[i].push_back(inputTensor);
    }


    /*Get the output tensors of the current subgraph*/
    std::vector<const xir::Tensor*> outputTensors = dpuRunner -> get_output_tensors();

    for(auto j: outputTensors) {

      std::vector<int> outputDimList;

      outputDimList.push_back(numFrames);

      const std::vector<int> outputDim = j -> get_shape();

      /*Get the height, width and batch size of the output tensor,
        the first value is the number of frames, which we already passed*/
      for(auto dim = outputDim.begin() + 1; dim != outputDim.end(); dim++)
        outputDimList.push_back(*dim);

      /*List with all the output tensors of the current subgraph*/
      std::vector<xt::xarray<int>> outputTensor(outputDimList[0], xt::zeros<int>({outputDimList[3], outputDimList[1], 
                                                                                  outputDimList[2]}));

      outputTensorList[i].push_back(outputTensor);
    }    

  }

}



void process_dpu_subgraph(vart::Runner *runner, int start, int end, std::vector<std::vector<xt::xarray<int>>> inputImages,
                          std::vector<std::vector<xt::xarray<int>>> outputImages, unsigned int subgraphID) {

  /*Get the input and output tensors of the DPU subgraph*/
  auto inputTensors = runner -> get_input_tensors();
  auto outputTensors = runner -> get_output_tensors();

  /*List with the dimensions: height, width and channel of every I/O tensor*/
  std::vector<std::vector<int>> inDimList;
  inDimList.reserve(inputTensors.size());

  std::vector<std::vector<int>> outDimList;
  outDimList.reserve(outputTensors.size());

  /*Calculate the dimensions and pass them to the list*/
  for(int i = 0; i < inputTensors.size(); i++)
    inDimList.push_back(get_tensor_dimensions(inputTensors[i]));

  for(int i = 0; i < outputTensors.size(); i++)
    outDimList.push_back(get_tensor_dimensions(outputTensors[i]));

  /*List with the size of every tensor => size = height * width * channel*/
  std::vector<std::size_t> inSizeList;
  inSizeList.reserve(inputTensors.size());

  std::vector<std::size_t> outSizeList;
  outSizeList.reserve(outputTensors.size());

  /*Calculate the size and pass them to the list*/
  for(int i = 0; i < inputTensors.size(); i++)
    inSizeList.push_back(get_tensor_size(inDimList[i]));

  for(int i = 0; i < outputTensors.size(); i++)
    outSizeList.push_back(get_tensor_size(outDimList[i]));
  



  /*I/O tensors which are passed to the DPU for processing*/
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs;
  std::vector<std::unique_ptr<vart::TensorBuffer>> outputs;

  std::vector<vart::TensorBuffer*> inputsPtr;
  std::vector<vart::TensorBuffer*> outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  std::vector<std::uint8_t *> inBatchList;
  inBatchList.reserve(inputTensors.size());

  std::vector<std::uint8_t *> outBatchList;
  outBatchList.reserve(outputTensors.size());

  int count = start;

  /*Iterate through every frame*/
  while(count < end) {

    for(auto inSize: inSizeList)
      inBatchList.push_back(new std::uint8_t[inSize]);

    for(auto outSize: outSizeList)
      outBatchList.push_back(new std::uint8_t[outSize]);


    /*Pass the values of every input image to the input batches, there is one batch for every input tensor*/
    for(int batch = 0; batch < inBatchList.size(); batch++)
      for(int height = 0; height < inDimList[batch][0]; height++)
        for(int width = 0; width < inDimList[batch][1]; width++)
          for(int channel = 0; channel < inDimList[batch][2]; channel++)
            inBatchList[batch][(height * inDimList[batch][1] + width) * inDimList[batch][2] + channel] = 
              inputImages[batch][count].at(channel, height, width);


    /* in/out tensor refactory for batch inout/output */
    for(int i = 0; i < inputTensors.size(); i++) {

      batchTensors.push_back(std::shared_ptr<xir::Tensor>(
          xir::Tensor::create(inputTensors[i] -> get_name(), inputTensors[i] -> get_shape(),
                              xir::DataType{xir::DataType::XINT, 8u})));

      inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
          inBatchList[i], batchTensors.back().get()));
    }

    for(int i = 0; i < outputTensors.size(); i++) {
    
      batchTensors.push_back(std::shared_ptr<xir::Tensor>(
          xir::Tensor::create(outputTensors[i] -> get_name(), outputTensors[i] -> get_shape(),
                              xir::DataType{xir::DataType::XINT, 8u})));

      outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
          outBatchList[i], batchTensors.back().get()));

    }

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();

    for(int i = 0; i < inputTensors.size(); i++)
      inputsPtr.push_back(inputs[i].get());

    for(int i = 0; i < outputTensors.size(); i++)
      outputsPtr.push_back(outputs[i].get());

    auto jobID = runner -> execute_async(inputsPtr, outputsPtr);
    runner -> wait(jobID.first, -1);

    
    /*Pass the values of every input image to the input batches, there is one batch for every input tensor*/
    for(int batch = 0; batch < outBatchList.size(); batch++)
      for(int height = 0; height < outDimList[batch][0]; height++)
        for(int width = 0; width < outDimList[batch][1]; width++)
          for(int channel = 0; channel < outDimList[batch][2]; channel++)
              outputImages[batch][count].at(channel, height, width) = 
              inBatchList[batch][(height * outDimList[batch][1] + width) * outDimList[batch][2] + channel];

    inputs.clear();
    outputs.clear();
    batchTensors.clear();

    for(auto inBatch: inBatchList)
      delete[] inBatch;

    for(auto outBatch: outBatchList)
      delete[] outBatch;

    inBatchList.clear();
    outBatchList.clear();

    count += 1;
  }

}


/*Process the UNet version*/
void run_unet(const std::string backbone, const unsigned int numThreads, std::vector<std::string> fullImageDirPath,
              std::vector<const xir::Subgraph*> dpuSubgraphs, std::vector<std::vector<std::vector<xt::xarray<int>>>> &inputTensorList,
              std::vector<std::vector<std::vector<xt::xarray<int>>>> &outputTensorList) {

  /*Start and end index for every batch, that is processed from different threads*/
  std::vector<std::uint8_t> startIndex;
  std::vector<std::uint8_t> endIndex;

  std::uint8_t start = 0;
  std::uint8_t end = 0;

  /*Divide the frames in batches to be processed from every thread, add the startIndex and endIndex
    of every batch to the lists*/
  for(unsigned int i = 1; i <= fullImageDirPath.size(); i++) {

    if(i == numThreads)
      end = fullImageDirPath.size() - 1;

    else
      end = std::floor((double)start + ((double)fullImageDirPath.size() / numThreads) - 1);


    startIndex.push_back(start);
    endIndex.push_back(end);

    start = end;
  }


  /*Check the backbone of the UNet, different backbone results in different number of subgraphs*/
  if(backbone == "resnet50") {

    /************************************************
     * CPU Subgraph 0
     ************************************************/

    preprocess_image(fullImageDirPath, inputTensorList[0][0]);

    /************************************************
     * DPU Subgraph 0
     ************************************************/

    unsigned int subgraphID = 0;

    std::vector<std::vector<xt::xarray<int>>> inputImages;
    inputImages.push_back(inputTensorList[subgraphID][0]);

    std::vector<std::vector<xt::xarray<int>>> outputImages;
    outputImages.push_back(outputTensorList[subgraphID][0]);

    std::vector<std::unique_ptr<vart::Runner>> dpuRunnerList;

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }



    std::vector<std::thread> threadList;

    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();

    /************************************************
     * CPU Subgraph 1
     ************************************************/

     cpu_zero_pad(outputTensorList[0][0], inputTensorList[1][0]);


    /************************************************
     * DPU Subgraph 1
     ************************************************/

    subgraphID = 1;

    inputImages.push_back(outputTensorList[0][0]);
    inputImages.push_back(inputTensorList[subgraphID][0]);
    outputImages.push_back(outputTensorList[subgraphID][0]);

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }


    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();

    /************************************************
     * CPU Subgraph 2
     ************************************************/

    std::vector<xt::xarray<int>> result = sigmoid_cpu_calc(outputTensorList[1][0]);

  }

  /*Process the UNet with mobilenetv2 backbone*/
  else {


    /************************************************
     * CPU Subgraph 0
     ************************************************/

    preprocess_image(fullImageDirPath, inputTensorList[0][0]);


    /************************************************
     * DPU Subgraph 0
     ************************************************/

    unsigned int subgraphID = 0;

    std::vector<std::vector<xt::xarray<int>>> inputImages;
    inputImages.push_back(inputTensorList[subgraphID][0]);

    std::vector<std::vector<xt::xarray<int>>> outputImages;
    outputImages.push_back(outputTensorList[subgraphID][0]);

    std::vector<std::unique_ptr<vart::Runner>> dpuRunnerList;

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }



    std::vector<std::thread> threadList;

    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();


    /************************************************
     * CPU Subgraph 1
     ************************************************/

    cpu_zero_pad(outputTensorList[0][0], inputTensorList[1][0]);


    /************************************************
     * DPU Subgraph 1
     ************************************************/

    subgraphID = 1;


    inputImages.push_back(inputTensorList[subgraphID][0]);

    outputImages.push_back(outputTensorList[subgraphID][0]);

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }


    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();


    /************************************************
     * CPU Subgraph 2
     ************************************************/

    cpu_zero_pad(outputTensorList[1][0], inputTensorList[2][1]);


    /************************************************
     * DPU Subgraph 2
     ************************************************/


    subgraphID = 2;


    inputImages.push_back(outputTensorList[1][0]);
    inputImages.push_back(inputTensorList[2][1]);

    outputImages.push_back(outputTensorList[subgraphID][0]);


    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }


    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();


    /************************************************
     * CPU Subgraph 3
     ************************************************/

    cpu_zero_pad(outputTensorList[2][0], inputTensorList[3][1]);


    /************************************************
     * DPU Subgraph 3
     ************************************************/

    subgraphID = 3;

    inputImages.push_back(outputTensorList[2][0]);
    inputImages.push_back(inputTensorList[3][1]);

    outputImages.push_back(outputTensorList[subgraphID][0]);

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }


    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();


    /************************************************
     * CPU Subgraph 4
     ************************************************/


    cpu_zero_pad(outputTensorList[3][0], inputTensorList[4][4]);

    /************************************************
     * DPU Subgraph 4
     ************************************************/

    subgraphID = 4;

    inputImages.push_back(outputTensorList[0][0]);
    inputImages.push_back(inputTensorList[1][0]);
    inputImages.push_back(outputTensorList[2][0]);
    inputImages.push_back(outputTensorList[3][0]);
    inputImages.push_back(inputTensorList[4][4]);


    outputImages.push_back(outputTensorList[subgraphID][0]);

    /*Assign a dpuRunner to every thread*/
    for(unsigned int i = 0; i < numThreads; i++) {

      auto dpuRunner = vart::Runner::create_runner(dpuSubgraphs[subgraphID], "run");
      dpuRunnerList.push_back(std::move(dpuRunner));
    }


    for(unsigned int i = 0; i < numThreads; i++)
      threadList.push_back(std::thread(process_dpu_subgraph, dpuRunnerList[i].get(), startIndex[i], endIndex[i], inputImages,
                                       outputImages, subgraphID));

    for(auto &thread: threadList)
      if(thread.joinable())
        thread.join();


    dpuRunnerList.clear();
    threadList.clear();
    inputImages.clear();
    outputImages.clear();


    /************************************************
     * CPU Subgraph 5
     ************************************************/

    std::vector<xt::xarray<int>> result = sigmoid_cpu_calc(outputTensorList[4][0]);

  }


}


/*Write the metrics numThreads, fps and runtime to a csv file with the name: metrics_backbone.csv*/
void write_csv(std::string backbone, std::vector<int> fpsList, std::vector<std::chrono::milliseconds> runtimeList,
                unsigned int numThreads) {

  std::string filename;

  filename.append("metrics").append("_").append(backbone).append(".csv");

  std::ofstream csvFile(filename);

  csvFile << "numThreads,fps,runtime" << "\n";

  for(unsigned int i = 1; i <= numThreads; i++) {

    double runtime = (double)(runtimeList[i - 1].count()) / 1000;
    csvFile << i << "," << fpsList[i - 1] << "," << runtime << "\n";
  }


  csvFile.close();
}
