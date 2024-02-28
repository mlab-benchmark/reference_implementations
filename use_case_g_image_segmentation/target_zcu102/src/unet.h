#ifndef UNET
#define UNET



#include <glog/logging.h>

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

/* header file for Vitis AI unified API */
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xadapt.hpp"



#define HEIGHT 224
#define WIDTH 224
#define CHANNEL 3

#define RESNET50 2
#define MOBILENETV2 4



/*Transforming an array of data to a xir::Tensor*/
class CpuFlatTensorBuffer : public vart::TensorBuffer {

 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : TensorBuffer{tensor}, data_{data} {}
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
    uint32_t size = std::ceil(tensor_->get_data_type().bit_width / 8.f);
    if (idx.size() == 0) {
      return {reinterpret_cast<uint64_t>(data_),
              tensor_->get_element_num() * size};
    }
    auto dims = tensor_->get_shape();
    auto offset = 0;
    for (auto k = 0; k < tensor_->get_shape().size(); k++) {
      auto stride = 1;
      for (auto m = k + 1; m < tensor_->get_shape().size(); m++) {
        stride *= dims[m];
      }
      offset += idx[k] * stride;
    }
    auto elem_num = tensor_->get_element_num();
    return {reinterpret_cast<uint64_t>(data_) + offset * size,
            (elem_num - offset) * size};
  }

 private:
  void* data_;
};



void print_data(std::vector<const xir::Subgraph*> dpuSubgraphs, std::vector<const xir::Subgraph*> cpuSubgraphs);

void list_images(std::string const& imageDirPath, std::vector<std::string>& imageList, unsigned int batchSize);

/*Get the subgraphs of the processingUnit: DPU or CPU*/
inline std::vector<const xir::Subgraph*> get_subgraph(const xir::Graph* graph, const std::string processingUnit) {

  auto root = graph -> get_root_subgraph();
  auto children = root -> children_topological_sort();

  auto ret = std::vector<const xir::Subgraph*>();

  /*Iterate through the subgraphs*/
  for(auto c: children) {

    CHECK(c -> has_attr("device"));

    /*Read the processingUnit, where the subgraphs is to be processed*/
    auto device = c -> get_attr<std::string>("device");

    /*Check if the device is DPU or CPU and put it in the vector*/
    if(device == processingUnit)
      ret.emplace_back(c);

  }

  return ret;

}


void preprocess_image(std::vector<std::string> fullImageDirPath, std::vector<xt::xarray<int>> &outputImages);

void run_unet(const std::string backbone, const unsigned int numThreads, std::vector<std::string> fullImageDirPath,
              std::vector<const xir::Subgraph*> dpuSubgraphs, std::vector<std::vector<std::vector<xt::xarray<int>>>> &inputTensorList,
              std::vector<std::vector<std::vector<xt::xarray<int>>>> &outputTensorList);


void write_csv(std::string backbone, std::vector<int> fpsList, std::vector<std::chrono::milliseconds> runtimeList,
                unsigned int numThreads);

/*Add the arrays of every I/O tensor to the corresponding subgraph and initialize them with zeros*/
void prepare_data_dpu(std::vector<const xir::Subgraph*> dpuSubgraphs, unsigned int numFrames ,
                      std::vector<std::vector<std::vector<xt::xarray<int>>>> &inputTensorList,
                      std::vector<std::vector<std::vector<xt::xarray<int>>>> &outputTensorList);

/*Implementation of 2D_Padding layer, that should run in the CPU*/
void cpu_zero_pad(std::vector<xt::xarray<int>> &inputImages, std::vector<xt::xarray<int>> &outputImages);

std::vector<xt::xarray<int>> sigmoid_cpu_calc(std::vector<xt::xarray<int>> images);

void process_dpu_subgraph(vart::Runner *runner, int start, int end, std::vector<std::vector<xt::xarray<int>>> inputImages,
                          std::vector<std::vector<xt::xarray<int>>> outputImages, unsigned int subgraphID);


/*Read the dimensions of a xir::Tensor and return them as a vector*/
std::vector<int> get_tensor_dimensions(const xir::Tensor* tensor);

/*Read the dimensions of a xir::Tensor and return them as a vector*/
std::size_t get_tensor_size(std::vector<int> dimList);


#endif