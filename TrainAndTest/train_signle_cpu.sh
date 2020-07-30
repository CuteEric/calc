#!/bin/bash

log_dir=train_logs
mkdir -p $log_dir

# Change to your caffe executable location. This one is standard for caffe cmake build in ~
caffe_exec=~/caffe/build/tools/caffe

# 2>&1 是将标准出错重定向到标准输出
$caffe_exec train --solver proto/solver.prototxt 2>&1 | tee $log_dir/`date +%m-%d-%y`.trainlog