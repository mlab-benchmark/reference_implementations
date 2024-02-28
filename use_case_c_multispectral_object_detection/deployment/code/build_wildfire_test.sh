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


# airplane automobile bird cat deer dog frog horse ship truck

tar -xvf wildfire_test.tar.xz &> /dev/null

cd ./wildfire_test

cd fire
mv *.png ../
cd ..
rm -r fire/

cd nofire
mv *.png ../
cd ..
rm -r nofire/
