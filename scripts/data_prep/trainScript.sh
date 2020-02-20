#!/bin/bash

dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
modelsRootDir="${FLIR_MODELS_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"

if [ -z "$dataRootDir" ]; then
    echo "Need to setup ML_TEST_DATA_SET_ROOT. Cannot Proceed Further" 2>&1 
    exit
fi

## Not Needed for training. Model being trained can be anywhere
if [ -z "$modelsRootDir" ]; then
    echo "Need to setup FLIR_MODELS_ROOT. Cannot Proceed Further" 2>&1 
    exit
fi

if [ -z "$caffeRoot" ]; then
    echo "Need to setup CAFFE_ROOT. Cannot Proceed Further" 2>&1 
    exit
fi
if [ -z "$cocoRoot" ]; then
    echo "Need to setup COCO_ROOT. Cannot Proceed Further" 2>&1 
    exit
fi


# default log file
lfn="/tmp/ktv.log"        

# Setup some default values (for testing )
# TODO: Need to get these as input arguments
#dataSetDir="ttaerial"
dataSetDir="vtest"
trainingDataSetDir="$dataRootDir""/""$dataSetDir"

# need to get these also as input argument
#modelFileSrcDir="object_detectors/train"
#modelFileName="train_foo_bar.caffemodel"
trainModelFileSrcDir="/home/flir/warea/train/syncity-experiments/drones/ssd_inception-v1_300x300"
preTrainModelFileName="inceptionV1_coco_SSD_300x300_iter_360000.caffemodel"

# this is the executable binary file and not a script
#tvScriptName="ssd_coco_512_syncityTest.py"

# other input args defaults here
dataSetType=0           # (1) test dataset (0) train_val
skipCreateLMDB=0        # skip LMDB creation (1) or not (0)
valSplitRatio=20       # percentage of test images from dataset

#### NOT NEEDED : START #########
catId=0        # ALL
#catId=3         # car
#catId=89        # drone : old
#catId=77        # parrot drone
#catId=78        # s1000 drone
#catId=79        # phantom drone
viewImages=0            # view detection images(1) or not (0)
numberOfImagesToView=2  # Number of Images is also settables
conf_thr=0.4
batch_size=13       # images batch size for detection
ap_ar=-1              # coco scores: -1 = All, 1 = AP, 0 = AR
#ap_ar=0              # coco scores: -1 = All, 1 = AP, 0 = AR
#ap_ar=1              # coco scores: -1 = All, 1 = AP, 0 = AR
area='all'          #all, small, medium, large
iouThr=0          #0:all, 0.5, 0.75
#### NOT NEEDED : END #########

# ==============================================================
# ALL Defaults should have been defined above
#
# Check for text "change to your needs" to modify per your needs
# ===============================================================

# setup dataSet Type String based on data type
if [ "$dataSetType" == 0 ]; then
    dataTypeStrVal="val"
    dataTypeStrTrain="train"
fi

if [ "$dataSetType" != 0 ]; then
    echo "dataSet Type $dataSetType NOT supported yet." 2>&1 
    exit
fi

# The labelmap_coco.prototxt & coco_labels.json are used when creating the LMDBs. 
# These should have the right label ID(s) for the object(s) in the dataset 
# the CNN will be trained with i.e., training/validation dataset
# Note: These will be looked for in the dir of the model being trained
coco_labelMapPrototxtFile="$trainModelFileSrcDir""/""labelmap_coco.prototxt"
coco_labelsJsonFile="$trainModelFileSrcDir""/""coco_labels.json"

#rm $lfn
touch $lfn 
echo "======================== START ======================" >> $lfn
date >> $lfn

#
# check and fail on environment variables here itself
# get the required input arguments, so that we can call other scripts.
# 

##########################################
#testDir=$1
#if [ -z "$testDir" ]; then
#    echo "Need Input Arguments to test specified model file with the validation dataset"
#    echo "using the following defaults instead: "
#    echo "dataSetDir                                   : $ML_TEST_DATA_SET_ROOT/$dataSetDir "
#    echo "dataSetType(1: validation, 0: train_test)    : $dataSetType"
#    echo "modelFileSrcDir                              : $FLIR_MODELS_ROOT/$modelFileSrcDir "
#    echo "modelFileName                                : $modelFileSrcDir/$modelFileName "
#    echo "detectionEvalScriptName                      : $detectionEvalScriptName "
#    echo "categoryId_to_check_results (o: all)         : $catId"
#    echo "skipCreateGT (1:yes, 0: No)                  : $skipCreateGT"
#    echo "skipRunDetEval (1:yes, 0: No)                : $skipRunDetEval"
#    echo "viewImages(1:yes, 0: No)                     : $viewImages"
#    echo "numberOfImagesToView                         : $numberOfImagesToView"
#    echo "confidence_threshold (10% ... 95%)           : $conf_thr"
#    echo "batch_size(1..512)                           : $batch_size"
#    echo "ap_ar (1:AP, 0: AR, -1: ALL)                 : $ap_ar"
#    echo "area (small, medium, large, all)             : $area"
#    echo "iouThr (0: ALL, 0.5, 0.75)                   : $iouThr"
#    echo "labelMapFile                                 : $labelMapFile"
#    echo "logFile (tail -f logFile for progress)       : $lfn"
#    echo " "
#    echo "NOTE: {CAFFE,COCO,FLIR_DATA,FLIR_MODELS}_ROOT & PYTHONPATH must be set properly"
#fi


echo "data root: $dataRootDir" 2>&1 | tee >> $lfn
echo "Caffe Root: $caffeRoot"  2>&1 | tee >> $lfn
echo "Coco Root: $cocoRoot"  2>&1 | tee >> $lfn
echo "log File: $lfn"  2>&1 | tee >> $lfn
echo "coco labelmap prototxt file: $coco_labelMapPrototxtFile" 2>&1 | tee >> $lfn
echo "coco labels json file: $coco_labelsJsonFile" 2>&1 | tee >> $lfn
echo "trainingDataSetDir: $trainingDataSetDir " 2>&1 | tee >> $lfn
echo "dataSetType(1: test, 0: train_val): $dataSetType" 2>&1 | tee >> $lfn
echo "trainModelFileSrcDir: $trainModelFileSrcDir " 2>&1 | tee >> $lfn
echo "preTrainModelFileName: $trainModelFileSrcDir/$preTrainModelFileName " 2>&1 | tee >> $lfn
##########################################

cd $caffeRoot

echo " ==0=================================================================== " 2>&1 | tee >> $lfn

#
# The Following needs to be called in sequence
#

myDir=$caffeRoot/scripts/data_prep
cd $myDir

if [ "$skipCreateLMDB" != 1 ]; then
    ## Note: the script prefixes the dataRootDir to the passed dataSetDir
    echo "python create_list_split.py $dataSetDir $coco_labelsJsonFile $valSplitRatio" 2>&1 | tee >> $lfn
    python create_list_split.py $dataSetDir $coco_labelsJsonFile $valSplitRatio 2>&1 | tee >> $lfn

    echo " ==1========================================================= " 2>&1 | tee >> $lfn 

    ## Note: the script prefixes the dataRootDir to the passed dataSetDir
    # The labelmap_coco.prototxt File - this is used by caffe_pb2..... 
    echo "./create_data_split.sh $dataSetDir $coco_labelMapPrototxtFile" 2>&1 | tee >> $lfn
    ./create_data_split.sh $dataSetDir $coco_labelMapPrototxtFile 2>&1 | tee >> $lfn
fi

echo " ==2============================================================== " 2>&1 | tee >> $lfn 

cd $caffeRoot
#myDir=$caffeRoot/build/examples/ssd

# Setup the script for training
##python scripts/data_prep/ssd_coco_eval_generic.py val_test models/VGGNet/coco/SSD_512x512 1 2>&1 | tee $lfn 
##echo "python "$modelFileSrcDir/$detectionEvalScriptName" $dataSetDir $modelFileSrcDir $dataSetType 2>&1 | tee $lfn "
#
#modelFile2Test="$modelsRootDir""/""$modelFileSrcDir""/""$modelFileName"
#trainResultsFile="$modelsRootDir""/""$modelFileSrcDir""/""$dataSetDir""_""$dataTypeStrtrain""_""results.txt"
#solverProtoFile="$modelsRootDir""/""$modelFileSrcDir""/""solver.prototxt"
#imageListFileTest="$dataRootDir""/""$dataSetDir""/""$dataTypeStrVal""/""$dataTypeStrVal""ImageList.txt"
#imageListFileTrain="$dataRootDir""/""$dataSetDir""/""$dataTypeStrTrain""/""$dataTypeStrTrain""ImageList.txt"
#resumeTraining=0       ## 1 is yes to resume training
#
#FLAGS="-confidence_threshold=""$conf_thr"" -view_image=0 -out_file=""$detResultsFile -batch_size=$batch_size" 
#
#
echo " ==x============================================================== " 2>&1 | tee >> $lfn

#echo "Bundling results under models directory into a single file." 2>&1 | tee >> $lfn
#myDir="$modelsRootDir""/""$modelFileSrcDir"
#cd $myDir
#
## change to your needs : backup directory & result files being tar & zipped
#backupDir="/tmp"         
#tarFileName="$dataSetDir""_""$dataTypeStr""_""catId""_""$catId"".tgz"
#filesToBeZipped="$dataSetDir""*.json ""$dataSetDir""*.txt "" *.log "
#echo "tar czvf $tarFileName $filesToBeZipped" 2>&1 | tee >> $lfn
#tar czvf $tarFileName $filesToBeZipped 2>&1 | tee >> $lfn
##tar --list -f $tarFileName 2>&1 | tee >> $lfn
##cp $tarFileName $backupDir 
#
## change this to your liking : to merge results*.json files into one big json file for other uses
## script to merge all json files only, to a single json file.
## python mergeJsonFiles.py $results_directory $out_filename.json
#
#
#echo " ==y============================================================== "  2>&1 | tee >> $lfn
#
#echo " DONE!! Check Results File & Logfile" 2>&1 | tee >> $lfn
#echo "======================== END ======================" >> $lfn
#

echo " DONE!! " 
#echo "  Results File $tarFileName " 
echo "  Logfile $lfn" 




