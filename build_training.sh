#!/bin/bash
set -v -e
NUM_PROC=$(grep -c ^processor /proc/cpuinfo)

CAFFE_DIR=$PWD
PROTOBUF_INSTALL_DIR="$CAFFE_DIR/protobuf-3.1.0"

sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y libleveldb-dev libsnappy-dev libboost-all-dev libhdf5-serial-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y python-pip
sudo apt-get install -y python-tk
sudo apt-get install -y ffmpeg
sudo apt-get install -y wget
sudo -H pip install numpy
sudo -H pip install scipy
sudo -H pip install scikit-learn scikit-image
pip install protobuf==3.1.0.post1


# Build Protobuf version 3.1.0 locally
sudo apt-get update
sudo apt-get install -y autoconf automake libtool curl make g++ unzip

PROTOBUF_VERSION=""
if [ -e $PROTOBUF_INSTALL_DIR/bin/protoc ] ; then
  PROTOBUF_VERSION=$($PROTOBUF_INSTALL_DIR/bin/protoc --version)
fi

if [ "$PROTOBUF_VERSION" != "libprotoc 3.1.0" ] ; then

  echo "Building Protobuf 3.1.0"
  echo "Protobuf will install locally to: $PROTOBUF_INSTALL_DIR"

  if [ ! -d protobuf-3.1.0 ]; then
    wget https://github.com/protocolbuffers/protobuf/archive/v3.1.0.tar.gz
    tar xzvf v3.1.0.tar.gz
  fi

  cd protobuf-3.1.0

  ./autogen.sh
  ./configure --prefix=$PROTOBUF_INSTALL_DIR
  make -j$NUM_PROC
  sudo make install   #installs to CAFFE_ROOT/protobuf-3.1.0
  sudo make clean
  cd ..

  if [ -e v3.1.0.tar.gz ]; then
    rm v3.1.0.tar.gz
  fi

  echo "Done installing Protobuf-3.1.0"
fi


# Build Caffe, passing the local version of protobuf
echo "Building Caffe"

cd $CAFFE_DIR
mkdir -p build && cd build

cmake \
  -DPROTOBUF_LIBRARY="$PROTOBUF_INSTALL_DIR/lib/libprotobuf.so" \
  -DPROTOBUF_INCLUDE_DIR="$PROTOBUF_INSTALL_DIR/include" \
  -DCUDNN_ROOT="/usr/"  \
  -D CMAKE_CXX_FLAGS="-D_FORCE_INLINES" \
  -D CMAKE_BUILD_TYPE=Release \
  -DPROTOBUF_LIBRARY_DEBUG="$PROTOBUF_INSTALL_DIR/lib/libprotobuf.so" \
  -DPROTOBUF_PROTOC_EXECUTABLE="$PROTOBUF_INSTALL_DIR/bin/protoc" \
  -DWITH_PYTHON_LAYER=1 \
  ..

make -j$NUM_PROC

