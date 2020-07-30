#!/bin/bash

# Build a build directory in DeepLCD
cd DeepLCD

if [ -d 'build' ]; then
    rm -rf build
else
    mkdir build && cd build
fi

# now build DeepLCD
# This is an example of using Caffe_ROOT_DIR since caffe isn't in ~/caffe
# you can change it to your caffe install dir
cmake -DCaffe_ROOT_DIR=~/caffe .. && make