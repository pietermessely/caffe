#!/bin/bash
cd "$(dirname "$0")"
set -v -e

pip install numpy
mkdir -p build && cd build
CAFFE_BUILDER_CONFIG=$EXT_PKGS'\caffe-builder\build_v140_x64\libraries\caffe-builder-config.cmake'
echo "CAFFE_BUILDER_CONFIG=$CAFFE_BUILDER_CONFIG"
#Comment below is a reference to change the cuda arches caffe is built for
#cmake -G "Visual Studio 14 2015 Win64" -C $CAFFE_BUILDER_CONFIG -D CUDA_ARCH_NAME=Manual -D CUDA_ARCH_BIN="75"  ..
cmake -G "Visual Studio 14 2015 Win64" -C $CAFFE_BUILDER_CONFIG  ..
cmake --build . --config Release
cmake --build . --config Debug
