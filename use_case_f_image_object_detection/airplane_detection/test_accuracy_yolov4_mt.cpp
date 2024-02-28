/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>
extern int g_last_frame_id;

#include <unordered_map>

std::string model_name;
bool is_first = true;
using namespace std;
namespace vitis {
namespace ai {

std::unordered_map<std::string, int> file_name_to_id_map = {
	{"014de911-7810-4f7d-8967-3e5402209f4a.jpg", 1},
	{"0263270b-e3ee-41dc-aeef-43ff77e66d5b.jpg", 2},
	{"03f84930-e2be-4c19-9afc-0dc26d02538e.jpg", 3},
	{"0641acc3-c0b9-4f9d-b0ba-7ad18aa08864.jpg", 4},
	{"074737f4-7f59-4729-be5d-67f6f1d34668.jpg", 5},
	{"11928d1c-44c7-4d3d-b593-9cdfd5f6f367.jpg", 6},
	{"12210ad7-83f8-4b54-bb4b-e93f8ff6ac1f.jpg", 7},
	{"135fdc4c-6656-4176-9873-9f00c6918293.jpg", 8},
	{"140d04fd-dea7-4d46-bce2-e20f73e155da.jpg", 9},
	{"14436c8c-93ec-41af-9fbf-43a5f39f2b98.jpg", 10},
	{"1e7e0450-6eb3-479e-88c2-990abc8207fa.jpg", 11},
	{"20bfce20-fb8a-4e20-a4aa-57064add848b.jpg", 12},
	{"22291e0b-ebe2-4f3f-b53e-4e709179300a.jpg", 13},
	{"22457f2e-a740-4719-9512-056749695281.jpg", 14},
	{"2314c1b5-ec8f-4212-b42f-43365a13fd20.jpg", 15},
	{"27cd32ba-c86a-419f-b544-601dd67c5d36.jpg", 16},
	{"29324dbf-6043-4521-bd83-44a97ffc5281.jpg", 17},
	{"3463b88d-1e11-4c04-9868-e8c72d510556.jpg", 18},
	{"34ae857d-6e71-46b9-b694-d9e40fb093bc.jpg", 19},
	{"38ff8d64-3460-4f83-bf9c-383832aeba0d.jpg", 20},
	{"3da0b873-fdde-4faf-9a85-021248c7dacf.jpg", 21},
	{"3e321b8a-9504-45aa-82b4-16158e28e290.jpg", 22},
	{"42f35cf8-66ca-400b-8381-dacc6483bb56.jpg", 23},
	{"48ef8e15-a43c-406b-9d3c-e815164b96d1.jpg", 24},
	{"495b73c8-024f-46cc-b426-05e49bbe5074.jpg", 25},
	{"4a0821b7-3689-4b0e-9397-389254ea9a32.jpg", 26},
	{"4c9d2482-788c-4d68-a3d4-478b2367abce.jpg", 27},
	{"4ca65749-67e4-4457-b4e4-89a3f58ce24d.jpg", 28},
	{"4e8c95f3-bbb6-4e2f-920d-8d3b2bd1d29f.jpg", 29},
	{"4e9164aa-532e-4b76-bce4-060b090da357.jpg", 30},
	{"4f833867-273e-4d73-8bc3-cb2d9ceb54ef.jpg", 31},
	{"4fdabd34-a2fd-4f0a-bb48-01fe043f1499.jpg", 32},
	{"534ba32c-cde2-402b-9b47-1396cf0999e8.jpg", 33},
	{"54442e49-5271-42bf-a8fb-a2cab30cfd49.jpg", 34},
	{"54b4e42b-3667-4564-b8fa-c23122ca54d5.jpg", 35},
	{"56765c5f-0922-43b3-8d58-bf2bf5a1e727.jpg", 36},
	{"56e2d3d3-6b16-401f-a300-847272373df5.jpg", 37},
	{"576827bc-a94a-4611-8820-f3d56e969151.jpg", 38},
	{"57af3c0a-b5ae-4e4f-a7f9-6856be2f80e5.jpg", 39},
	{"58956bcc-11fc-4357-8d75-32fee9feaf07.jpg", 40},
	{"5bb8ab95-f141-43ca-a722-a1937d9d5a72.jpg", 41},
	{"5c9e817a-dc4b-42ab-952c-3128e2de12e8.jpg", 42},
	{"5f726a3d-9876-498f-90c5-9f9efae06c6a.jpg", 43},
	{"65e32857-71dc-406b-8583-1ea6467a330f.jpg", 44},
	{"6627e7c7-2fdd-4f3c-965e-b4d73d0a4cc2.jpg", 45},
	{"696b9320-7dbe-4c28-bd07-7a73e7a28e64.jpg", 46},
	{"70332141-fcd3-4e2e-89c7-d50cbc03c25f.jpg", 47},
	{"71252436-5176-40ba-b1ff-73efc251f3c6.jpg", 48},
	{"72fa2a77-cf8e-44a2-9ab2-f11ca63b4b72.jpg", 49},
	{"74d335db-bf64-424a-88d7-5c24625c50b1.jpg", 50},
	{"7635d63c-6b97-4c9c-a7dc-27773d42ed4c.jpg", 51},
	{"77f7b57f-5cf2-424d-a952-9847b3c3f35e.jpg", 52},
	{"78099b50-f2b6-4319-b462-f33df2966c45.jpg", 53},
	{"78400c58-1a7c-4342-a1fb-2117cb7cbc8b.jpg", 54},
	{"791377f0-a01c-4fc3-b6e9-151b6cb80fdf.jpg", 55},
	{"7ab2461e-fad4-4d90-a260-8317bbc4e958.jpg", 56},
	{"7c2da441-819f-4ddd-8293-f4be5213d69e.jpg", 57},
	{"8df07836-4606-446e-9880-6ed9e0f74543.jpg", 58},
	{"9089631b-b1c0-4df8-b3f4-061c98149a94.jpg", 59},
	{"90c365f8-18b6-4230-be74-cd856a1ba98f.jpg", 60},
	{"980b5831-43b7-4adb-9e4b-67d6cff3ef68.jpg", 61},
	{"996ba764-cd2a-49c0-9c9f-b166207de0ab.jpg", 62},
	{"a32efaae-a263-4570-903f-76cf9942742f.jpg", 63},
	{"a40b7aed-8db8-449d-b0e4-3debfa04281a.jpg", 64},
	{"a9cc0c57-a46a-4d85-bd97-937cf7ffc6f8.jpg", 65},
	{"ac77b86a-4198-4d40-900e-a95e520b2ac7.jpg", 66},
	{"ad36e699-4a1f-4891-b579-6e9ac3e54235.jpg", 67},
	{"ae2fddcd-a2b4-4f32-b86d-ac2c2b03d77d.jpg", 68},
	{"af67041b-f363-47ae-8ddd-f652db3a6bab.jpg", 69},
	{"b3324b5e-cca9-42f7-9806-ab1f86eee050.jpg", 70},
	{"b3c7c22d-9c71-4ff3-9146-e2f9ab2f9787.jpg", 71},
	{"b6e9f8fc-3fbd-411a-a1e5-b853e8fbe2ac.jpg", 72},
	{"b7ab0316-bf02-4266-b44a-cef6417f795c.jpg", 73},
	{"babb0ef2-ef2d-4cab-b3e2-230ae2418cdc.jpg", 74},
	{"bd4fba16-ec59-4163-a896-db2274b86a8b.jpg", 75},
	{"bed88cf9-7887-4bfd-a1d7-b5499ee24f03.jpg", 76},
	{"bfc53eb0-eb57-4dd7-8c1e-8e42ba49757b.jpg", 77},
	{"c20b3c21-a6e9-46f7-8536-d4f92574e111.jpg", 78},
	{"c98a544c-4206-4176-b480-35b8bdf5bb14.jpg", 79},
	{"ca7b1077-0d37-4ada-9d56-0e66f864935e.jpg", 80},
	{"cbd51501-ed0f-411c-b472-df4357cca40c.jpg", 81},
	{"cc4f3226-c262-409e-a4b2-a576e776f7f4.jpg", 82},
	{"d0c3d270-f23e-4792-bac0-142a9cc8ccc6.jpg", 83},
	{"d3d2b706-9017-41f4-b57e-469038daa634.jpg", 84},
	{"d8873734-016a-4b9d-9b9e-8bc47eb13fef.jpg", 85},
	{"d9399a45-6745-4e59-8903-90640b2ddf9f.jpg", 86},
	{"dacabde6-c8a7-475b-ab26-81b4e4dd5977.jpg", 87},
	{"dce20c6a-d661-4258-9c58-5507f38ce97e.jpg", 88},
	{"e18dde99-6efd-4c6a-8fd3-343e8a393cd8.jpg", 89},
	{"e5416a97-a8ea-415f-928a-e75a5090cd46.jpg", 90},
	{"e9021b38-0b3c-484f-af2c-b36bbf765f85.jpg", 91},
	{"ecfe7982-05e5-435f-824b-e24b6846316e.jpg", 92},
	{"ed921ec7-b950-460c-a912-12247d867fea.jpg", 93},
	{"eeb3f271-15f5-4977-a921-f4636efc6434.jpg", 94},
	{"eeb978ec-5945-4def-819a-4ea903b17c2d.jpg", 95},
	{"ef8ff443-e09b-405d-a432-3b5bde0a278b.jpg", 96},
	{"ef92f434-c41b-4423-9218-4514c49f00c6.jpg", 97},
	{"f34f05c6-84ce-47d7-ba5c-394d85c86a46.jpg", 98},
	{"f82d64a6-3bfa-4612-bcbd-847a7d89c296.jpg", 99},
	{"fc06c595-7905-46c3-91c0-daf56af0f926.jpg", 100},
	{"fc1ab8ce-e531-46ed-b74b-0374cd58cf2a.jpg", 101},
	{"fc8f9dc5-b6a2-49f3-9bde-1f8ff382ca5f.jpg", 102},
	{"fca12ef2-0f65-48d0-a4aa-6bae562f2236.jpg", 103}
};

vector<int> coco_id_map_dict() {
  vector<int> category_ids;
  category_ids = {1};
  return category_ids;
}

int imagename_to_id(string imagename) {

  //int idx1 = imagename.size();
  //int idx2 = imagename.find_last_of('_');
  //string id = imagename.substr(idx2+1, idx1-idx2);
  //int image_id = atoi(id.c_str());

  int image_id = vitis::ai::file_name_to_id_map [imagename];
  return image_id;
}

struct Yolov3Acc : public AccThread {
  Yolov3Acc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~Yolov3Acc() { of.close(); }

  static std::shared_ptr<Yolov3Acc> instance(std::string output_file) {
    static std::weak_ptr<Yolov3Acc> the_instance;
    std::shared_ptr<Yolov3Acc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<Yolov3Acc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto ccoco_id_map_dict = coco_id_map_dict();
    auto result = (YOLOv3Result*)dpu_result.result_ptr.get();
    for (auto& box : result->bboxes) {
      float xmin = box.x * dpu_result.w;
      float ymin = box.y * dpu_result.h;
      float xmax = (box.x + box.width) * dpu_result.w;
      float ymax = (box.y + box.height) * dpu_result.h;
      if (xmin < 0) xmin = 1;
      if (ymin < 0) ymin = 1;
      if (xmax > dpu_result.w) xmax = dpu_result.w;
      if (ymax > dpu_result.h) ymax = dpu_result.h;
      float confidence = box.score;
      of << fixed << setprecision(0) <<"{\"image_id\":" <<  imagename_to_id(dpu_result.single_name) <<
      ", \"category_id\":" << ccoco_id_map_dict[box.label]<< ", \"bbox\":[" << fixed <<
      setprecision(6) << xmin << ", " << ymin << ", " << xmax-xmin << ", " << ymax-ymin << "], \"score\":"
      << confidence << "}," << endl;
    }
  }

  virtual int run() override {
    if(is_first) {
      of << "[" << endl; 
      is_first = false;
    }
    if (g_last_frame_id == int(dpu_result.frame_id)) {
      of.seekp(-2L, ios::end);
      of << endl << "]" << endl;  
      return -1;
    }
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000))){
      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {

  model_name = argv[1];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::YOLOv3::create(model_name); },
      //[&] { return vitis::ai::YOLOv3::create(model_name + "_acc"); },
      vitis::ai::Yolov3Acc::instance(argv[3]), 2);
}
