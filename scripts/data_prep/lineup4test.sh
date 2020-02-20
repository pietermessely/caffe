#!/bin/bash
#
##
#       Q4'2018 : kedar
#         extend to provide a max itertion limit to bailout.
#         Useful for automation to end the script
#
##
#       Update 8/27/18 by Evan
#         Changed default IoU threshold to 0.5
#         Updated docs
## ====================================================================================================================
#       Author: kedar
#       Date:   Q2'2018
#
#       This is useful to run evaluation on .caffemodel files in a directory.
#       The intention is to automate the evaluation of the training snapshots while training to get
#           an idea of how the training datasets are helping or not. This can be automated to either stop the
#           training or change/augment the training datasets, etc.,
#       The script repeatedly calls testScript.sh for each model file in modelFileSrc dir.
#       All environment variables testScript.sh uses must be set properly.
#       It takes arguments as following:
#           a. Test-dataset directory (realtive to ML_TEST_DATA_ROOT)
#           b. modelFileSrc dir: expected to find the model files here (relative to FLIR_MODELS_ROOT)
#           c. Prefix of model file name up through "xxxxx_iter_"
#           d. Start iteration number of the model to evaluate
#           e. Step value for iteration incrementation
#           f. Calculate map prime? (0 or 1)
#           g. Number of images to save
#           h. Maximum number of iterations after which the script should end
#           i. Category id to evaluate on
#           j. Batch size to use
#           k. Confidence threshold to use
#       Note: The IoU threshold is set to 0.5
#       Example command: bash lineup4test.sh dronedata object_detectors/drones/training_experiments/modelsBeingTuned/mar5-hn-mining-mcd-nogb inceptionV1_coco_mcd_mar5_nogb_SSD_512x512_iter_ 250 250 0 0 10000
#
## ====================================================================================================================

modelsRootDir="${FLIR_MODELS_ROOT:-""}"      # make sure FLIR_MODELS_ROOT is set properly for testing snapshot models
caffeRootDir="${CAFFE_ROOT:-""}"

dataSetDir=$1
modelFileSrcDir=$2
mfnPrefix=$3

iterNum=1000
iterNum=$4

iterStep=1000
iterStep=$5

mapPrime=0
mapPrime=$6

saveImages=0
saveImages=$7

maxIterNum=1000000
maxIterNum=$8

categoryNum=0
categoryNum=$9

batchSize=1
batchSize=${10}

confThr=0.01
confThr=${11}


# setup the log file name based on the modelFileSrcDir
lfn="/tmp/""$USER""/k.log"

fext=".caffemodel"
scriptFile="$CAFFE_ROOT""/scripts/data_prep/testScript.sh"

Date=`date`
echo "============================== $Date =========================== "
echo "=========================== $Date ======================== " 2>&1|tee>>/tmp/k.log

echo "== starting Iteration: $iterNum,  iteration step: $iterStep, maxIter: $maxiterNum"

while true; do

    mfn="$mfnPrefix""$iterNum""$fext"
    mfnFullPath="$FLIR_MODELS_ROOT""/""$modelFileSrcDir""/""$mfnPrefix""$iterNum""$fext"
    echo "$mfnFullPath"

    if [ -f "$mfnFullPath" ]; then
        echo "== Running performance eval for model $mfn"
        echo "$scriptFile -d $dataSetDir -m $modelFileSrcDir -f $mfn -s 1 -i 0 -b $batchSize -e 0 -c $categoryNum -a $mapPrime -z $saveImages -y 0.5 -w $lfn -t $confThr"
        $scriptFile -d $dataSetDir -m $modelFileSrcDir -f $mfn -s 1 -i 0 -b $batchSize -e 0 -c $categoryNum -a $mapPrime -z $saveImages -y 0.5 -w $lfn -t $confThr
        iterNum=$(($iterNum + $iterStep))
    else
        # wait for a bit to see if get a snapshot from the training area
        echo "== $mfnFullPath Not available yet. Sleeping for a few minutes ... zzzzz"
        sleep 300
    fi

    if [ "$iterNum" -gt "$maxIterNum" ]; then
        echo " == Bailing Out: Maximum Number of iterations reached === $maxIterNum : $iterNum"
        break;
    fi


done
