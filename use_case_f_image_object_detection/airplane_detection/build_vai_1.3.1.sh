#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CXX=${CXX:-g++}
CFLAGS=-std=c++17
$CXX $CFLAGS -I. -o test_accuracy_yolov4 test_accuracy_yolov4_mt.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-yolov3  -lvitis_ai_library-model_config -lglog -lvitis_ai_library-dpu_task -lpthread
#$CXX $CFLAGS -I. -o test_jpeg_yolov4 test_jpeg_yolov4.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-yolov3  -pthread -lglog 
#$CXX $CFLAGS -I. -o test_performance_yolov4 test_performance_yolov4.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-yolov3  -lvart-util -lvitis_ai_library-dpu_task -pthread -lglog 
#$CXX $CFLAGS -I. -o test_video_yolov4 test_video_yolov4.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-yolov3  -pthread -lglog 
