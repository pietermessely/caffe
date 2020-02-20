#!/bin/bash

#
# Created this to simplify the dataset preparation.
# Setup GT file, imageSet File & name-size file based on environemnt variables and input arguments.
# Get the two input arguments : data-src-dir and use-of-data-set [validation(1) or Train_Test(0)]
# Call the actual python script in COCO_ROOT/PythonAPI/scripts/get_image_size.py with <args>
# 


#
# setup context from environment variables
# 

dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"
pyPath="${PYTHONPATH:-""}"

if [ -z "$dataRootDir" ]; then
    echo "Need to setup ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi

if [ -z "$caffeRoot" ]; then
    echo "Need to setup CAFFE_ROOT. Cannot Proceed Further"
    exit
fi

if [ -z "$cocoRoot" ]; then
    echo "Need to setup COCO_ROOT. Cannot Proceed Further"
    exit
fi

# if PYTHONPATH does not exist, setup based on CAFFE_ROOT and COCO_ROOT
if [ -z "$pyPath" ]; then
    echo "Setting up PYTHONPATH based on CAFFE_ROOT & COCO_ROOT"
    PYTHONPATH="${CAFFE_ROOT}/python:${COCO_ROOT}/PythonAPI:${CAFFE_ROOT}/scripts/data_prep"
    export PYTHONPATH="${PYTHONPATH}"
fi


# useful for debug
#echo "data root: $dataRootDir"
#echo "Caffe Root: $caffeRoot" 
#echo "Coco Root: $cocoRoot" 
#echo "PYTHONPATH: ${PYTHONPATH}" 

dataSrcDir=$1
if [ -z "$dataSrcDir" ]; then
    echo "Need to provide data Set Directory relative to ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi

dataType=$2
if [ -z "$dataType" ]; then
    echo "Need to provide data Type also i.e, 1:Validation or 0:Train_Test. Cannot Proceed Further"
    exit
fi

gtFile=$3
if [ ! -f "${gtFile}" ]; then
    echo "Need to provide Ground Truth File, Absolute Path"
    exit
fi

if [ "$dataType" == 1 ]; then
    dataTypeStr="test"
fi

if [ "$dataType" != 1 ]; then
    echo "dataType $dataType NOT supported"
    exit
fi


echo "data set src dir is $dataSrcDir,  dataType: $dataType ($dataTypeStr)"

pyScript="$cocoRoot/PythonAPI/scripts/get_image_size.py"
imageListFile="$dataRootDir/$dataSrcDir/$dataTypeStr/$dataTypeStr""ImageList.txt"
nameSizeFile="$dataRootDir/$dataSrcDir/$dataTypeStr/$dataTypeStr""_name_size.txt"

echo "Executing: python $pyScript $gtFile $imageListFile $nameSizeFile"
python $pyScript $gtFile $imageListFile $nameSizeFile


