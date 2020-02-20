#!/bin/bash
set -v -e
NUM_PROC=$(grep -c ^processor /proc/cpuinfo)
PROC_TYPE=$(arch)

sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y libleveldb-dev libsnappy-dev libboost-all-dev libhdf5-serial-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y python-pip
sudo apt-get install -y python-tk
sudo apt-get install -y ffmpeg


# I don't think we need to build pycaffe on the embedded platforms
if [ $PROC_TYPE != "aarch64" ]
then
  sudo -H pip2 install -r python2_requirements.txt
fi

if [[ ! -z $1 && $1 == "release-arches" ]]
then
  CUDA_ARCH_SELECTION=( -D CUDA_ARCH_NAME=Manual -D CUDA_ARCH_BIN="61 70 75" )
fi

mkdir -p build && cd build

cmake -DCUDNN_ROOT="/usr/" -D CMAKE_CXX_FLAGS="-D_FORCE_INLINES -std=c++11" "${CUDA_ARCH_SELECTION[@]}" -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/caffe .. || true

make -j$NUM_PROC

sudo make -j$NUM_PROC install

echo "Adding /usr/local/caffe/lib to ld config"
touch caffe.conf
echo "/usr/local/caffe/lib" > caffe.conf
sudo chmod 644 caffe.conf
sudo chown :root caffe.conf
sudo mv caffe.conf /etc/ld.so.conf.d/
sudo ldconfig
