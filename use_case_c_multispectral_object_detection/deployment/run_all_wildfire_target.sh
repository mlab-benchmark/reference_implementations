#!/bin/bash

## © Copyright (C) 2016-2020 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.

# build fmnist test images
bash ./code/build_wildfire_test.sh


## compile CNN application
# cd code
# bash -x ./build_app.sh
# mv code ../run_cnn # change name of the application
# bash -x ./build_get_dpu_fps.sh
# mv code ../get_dpu_fps
# cd ..

/run_cnn ./miniVggNet.xmodel       ./wildfire_test/ ./code/wildfire_labels.txt | tee ./rpt/logfile_wildfire_miniVggNet.txt

# check DPU prediction accuracy
bash -x ./code/check_wildfire_accuracy.sh | tee ./rpt/summary_wildfire_prediction_accuracy_on_dpu.txt

# run multithreading Python VART APIs to get fps
bash -x ./code/fps_wildfire.sh | tee ./rpt/logfile_wildfire_fps.txt

# remove images
rm -r wildfire_test
