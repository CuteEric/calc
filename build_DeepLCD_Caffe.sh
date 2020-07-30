#!/bin/bash

cd DeepLCD && mkdir build && cd build
# Build caffe in our build directory

#有很多bug，应该替换成最新版本的caffe
#git clone https://github.com/nmerrill67/caffe.git

# download newest caffe code
git clone https://github.com/BVLC/caffe

# to caff dir
cd caffe

# configure for CPU_ONLY
sed -i '/caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA/c\caffe_option(CPU_ONLY  "Build Caffe without CUDA support" ON) # TODO: rename to USE_CUDA' CMakeLists.txt
sed -i '1s/^/#define CPU_ONLY 1\n/' include/caffe/util/device_alternate.hpp
sed -i '/# CPU_ONLY := 1/c\CPU_ONLY := 1' Makefile.config

# make
mkdir build && cd build
cmake -DBLAS=open .. && make && sudo make install

# back to DeepLCD/build
cd ../..

# now build DeepLCD
cmake -DCaffe_ROOT_DIR=$PWD/caffe .. && make # This is an example of using Caffe_ROOT_DIR since caffe isn't in ~/caffe