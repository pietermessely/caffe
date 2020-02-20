#!/bin/bash

###############################################################################################
#
# Q2'2018: copied testScript.sh and modified / augmented as necessary
# Author:  kedar
#
# Purpose
#
#       Generate CNN performance evaluation metrics based on
#              detection results input file, against the requested test dataset.
#
#       At a glance, this script does the following:
#               Scoring per coco metrics & output format
#               Scoring per Flir specific needs: FP, FN, Detections Missed.
#               Scoring on a specific category or on all available categories.
#               Optionally, view detections on images
#               Generates results files and logs.
#               Creates one date_time stamped tgz file in the model_UT dir.
#
#       Input Args:
#               1. detection-results.txt file
#                       format of each line:
#                       full-path-of-image categoryId score xmin ymin xmax ymax time1 time2
#                       where, time1 & time2 are micro-secs time through CNN->forward &
#                       detections eval overhead of CNN, report as '0' if not available.
#                       Format of each element in the line is:
#                       string int float float float float float int int
#               2. absolute path to labelVector.json file
#               2. absolute path to labelMap.json file
#               3. relative path to test dataset dir from ML_TEST_DATA_SET_ROOT(need annotation_set.json file)
#               4. skip(1) or generate(0) GT: First time, you need to generate GT.
#
#       Environment Variables needed: CAFFE_ROOT, COCO_ROOT, PYTHONPATH, ML_TEST_DATA_SET_ROOT
#               where PYTHONPATH includes:
#               CAFFE_ROOT/python, CAFFE_ROOT/scripts/data_prep & COCO_ROOT/PythonAPI
#       e.g., ./evaluateDetections.sh \
#                 -f /mnt/fruitbasket/users/kmadineni/adas/cortex-test/dec12-adas.txt \
#                 -d t07_jul12 -l /mnt/data/models/object_detectors/labelfile_templates/labelvectors/flir_labelvector.json \
#                 -m /mnt/data/models/object_detectors/labelfile_templates/catids/flir_catids.json \
#                 -s 0 -y 1.0 -c 0
#                 NOTE: use "-h 0" if testdata dir is NOT setup as download from conservator. This way, script
#                       uses the filenames in the evaluations input file as is.
#
###############################################################################################

set -e


dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"
pyPath="${PYTHONPATH:-""}"

#echo "FLIR_MODELS_ROOT: $FLIR_MODELS_ROOT"
echo "ML_TEST_DATA_SET_ROOT: $ML_TEST_DATA_SET_ROOT"
echo "CAFFE_ROOT: $CAFFE_ROOT"
echo "COCO_ROOT: $COCO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

#if [ -z "$modelsRootDir" ]; then
#    echo "Need to setup FLIR_MODELS_ROOT. Cannot Proceed Further"
#    exit
#fi
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
if [ -z "$pyPath" ]; then
    echo "Need to setup PYTHONPATH. Cannot Proceed Further"
    exit
fi

##
# Setup some default values (for testing )
##

# default log file
if [ -z "$USER" ]; then
    USER=`whoami`
fi

logPrefixDir="/tmp/$USER"
lfn="$logPrefixDir/ke.log"

if [ ! -d "$logPrefixDir" ]; then
    echo "Creating $logPrefixDir" 2>&1
    mkdir $logPrefixDir 2>&1
fi

##
# Setup some default dir for saving logs and outputs
##
cnnEvalResultsDir="$logPrefixDir""/cnnEvalResults"
if [ ! -d "$cnnEvalResultsDir" ]; then
    ## create workspace dir for saving output videos & detection images
    echo "Creating $cnnEvalResultsDir" 2>&1
    mkdir $cnnEvalResultsDir 2>&1
fi

wsd="$cnnEvalResultsDir""/workspace"
if [ ! -d "$wsd" ]; then
    echo "Creating $wsd" 2>&1
    mkdir "$wsd" 2>&1
fi

# defaults for input arguments
dataSetDir="test_astoria-aerial"
dataSetType=1           # (1) Test dataset (0) train_val
dataTypeStr="test"      # only 'test' is supported in this script
catId=0                 # ALL classes (present in ground truth)

#Steps:
#  1. Convert all ground-truth to format that we need. ("CreateGT")
#  2. Evaluate input_detections_file.txt against GT using COCO metrics
#  3. Generate images for diagnosing CNN performance (draw detections and bounding boxes)
skipCreateGT=0          # 1: skip creating ground truth files

# These flag controls which metrics are computed. There are lots of metrics, so sometimes
# it is not convenient (and time consuming) to compute of them. So this lets you only compute
# a subset of the metrics.
ap_ar=-1              # coco scores: -1 = All, 1 = AP, 0 = AR
iouThr=0          #0:all, 0.5, 0.75
areaStr=("small" "medium" "large" "all")
area="all"          #all, small, medium, large
generate_mAP_prime=0

# max IoU Threshold to use when calculating the Precision & Recall values
# default value of 1 means the standard COCO metrics i.e., scores with IoU threshold from
#       0.5-0.95 with 0.5 increments above the confidence threshold will be used to
#       evaluate the mAP and mAR metrics.
maxIouThrArg="--maxIouThr 1.0"

## Generate data for plotting P-R and RoC curves.
#       0: No plots, 1: P-R Curve, 2: RoC curves, 3: both
#       confidence-threhold used for generating the data are: 0.3 through 0.9
plotPRArg="--plotPR 0"
plotPR=0
plotRoC=0
confThr=0.01
thrStr="dot0"

# By default disable creating test video. This will also disable saving output images
createVideo=0

# annotations file to use for creating the GT
imageJsonFileName="image_set.json"
imageJsonFileFullPath=""
annoJsonFileName="annotation_set.json"
annoJsonFileFullPath=""

# default: modify the image_name for each detection, in detections file
modifyImageNameInEachDetection=1

imageDirPrefixArg="--imageDirPrefix PreviewData"

# include all GT or remove images with no annotations
includeAllGtArg="--includeAllGt yes"

labelRemapArg=""

# ====================================================================
# ALL Defaults should have been defined above
#
# "change to your needs" identifies sections to modify per your needs
# ====================================================================


# ===================================================================================
# Read Input Arguments if any and update the variables
#
# Mandatory arguments are noted with *
#       If optional input args are present, SHOULD be in the FORMAT as noted below.
#       When NOT present, default values assigned above will be used
#
# Format of Input Args are:
#
#       -a: mAP Prime:
#               [0|resize] : 0 (default) disables calculating mAP-Prime. resize: enables this,
#                        using the value as the resize value to resize frame
#
#       -c : Category-ID:
#               [0..90], where '0' is ALL and 'x' is mapped coco ID.
#               This will be used to gather/calcuate the test results.
#
#       -d : test_data_set_directory relative to ML_TEST_DATA_SET_ROOT
#               e.g., input "mytestdataset" dir should be at "$ML_TEST_DATA_SET_ROOT/mytestdataset"
#
#       -f : Model_Detections_Output_File_Name:
#               model detections output file to be used. Absolute path.
#               "The format of the output in this file should be (each line):
#                absolute-path-of-image-filename categoryId score xmin ymin xmax ymax time1 time2
#                   where, time1&2 is time n micro-secs taken by CNN->forward and 'detections'
#                   processing respectively. If these are not available, set it to 0
#
#       -h : hack modify detection output image_name format in the detections_output file (option -f )
#               The first element in each is the absolute path to the image, where
#                   Image ID is part of the filename. ImageId is assigned based on index in
#                   the image_set.json file.
#               e.g., 0: do not change. 1: change per evaluation needs (default)
#
#       -j : images.json File_Name:
#               images file. Must be in "$mytestdataset" dir from '-d' option above
#
#       -l : Absolute path to labelvector.json file
#
#       -m : Absolute path to labelmap.json file
#
#       -p: plot ROC & PR Curves:
#               [0/1/2/3], where  0: No Plot, 1: ROC Plot, 2: PR Plot, 3: both
#
#       -r: include all GT or remove images with no annotations
#               "yes" : include all GT (default), "no" : remove images with no annotations
#
#       -s: Skip_GT:
#               [0/1] : 0 is don't skip and 1 is skip
#                       NOTE: you have to create the GT first time with this script. This is not
#                       compatible with the GT created using testScript.sh due to filename differences
#   
#       -t: confidence_threshold:
#               [0.01 .. 0.90] : 0.01 is the default, to compute metrics in the entire confidence threshold range.
#                                Note, time to compute metrics will be longer with default value
#
#       -w: Log File name to use. Full path.
#
#       -x: image directory prefix to grab the images from for GT:
#               PreviewData (default) or Data
#
#       -y: max IoU Thrshold to use for Precision and Recall metrics:
#               [0.5 - 1.0, steps of 0.5, where 1.0 defaults to standard range 0.5 - 0.95
#
#       -z: create detection output Video:
#               [0/1] : 0 will not create a video and 1 creates the video with test images.
#                       These images have both the GT and Detection bounding boxes with
#                       category ID & scores from the model under test.
#
#       e.g., ./evaluateDetections.sh \
#                 -f /mnt/fruitbasket/users/kmadineni/adas/cortex-test/dec12-adas.txt \
#                 -d t07_jul12 -l /mnt/data/models/object_detectors/labelfile_templates/labelvectors/flir_labelvector.json \
#                 -m /mnt/data/models/object_detectors/labelfile_templates/catids/flir_catids.json \
#                 -s 0 -y 1.0 -c 0 -h 0
#                 NOTE: use "-h 0" if testdata dir is NOT setup as download from conservator. This way, script
#                       uses the filenames in the evaluations input file as is.
#
# ===================================================================================

while getopts a:c:d:f:h:l:L:m:p:r:s:t:w:x:y:z: option
do
 case "${option}" in

    a) # generate map-prime also
        generate_mAP_prime=$OPTARG
        echo "Generate mAP Prime: $generate_mAP_prime"
        ;;

    c)  # category ID
        catId=${OPTARG}
        ;;

    d)  # dataset directory to be used
        dataSetDir=${OPTARG}
        ;;


    f)  # CNN detections output file to be evaluated. Full path
        detectionResultsInputFile=$(realpath $OPTARG)
        detResultsFile=$(realpath $OPTARG)
        ;;

    h)  ## hack i.e., change imageId (filename) element in detections_output file per evaluation script needs
        modifyImageNameInEachDetection=$OPTARG
        if [ $modifyImageNameInEachDetection -ge '1' ] ; then
            modifyImageNameInEachDetection=1
        else
            modifyImageNameInEachDetection=0
        fi
        ;;

    l)  # absolute path of the labels.json file to use for creating GT
        jsonLabelVectorFile=$OPTARG
        ;;

    L)  # label remapping json file
        labelRemap=$OPTARG
        ;;

    m)  # absolute path of the labels.json file to use for creating GT
        jsonLabelMapFile=$OPTARG
        ;;

    p)  # Plot requests
        plotReq=$OPTARG
        case $plotReq in
            0)
                plotRoC=0
                plotPR=0
                ;;
            1)
                plotRoC=1
                plotPR=0
                ;;
            2)
                plotRoC=0
                plotPR=1
                ;;
            3)
                plotRoC=1
                plotPR=1
                ;;
        esac
        plotPRArg="--plotPR $plotPR"
        ;;

    r)  # include_all_gt argument : grab and pass to createList.py
        includeAllGtArg="--includeAllGt $OPTARG"
        ;;

    s)  # skip creating GT from the data set being used for testing
        skipCreateGT=$OPTARG
        if [ $skipCreateGT -lt '0' ] || [ $skipCreateGT -gt '1' ] ; then
            skipCreateGT=0
            echo "changed skipCreateGT to: $skipCreateGT"
        fi
        ;;

    t)  # confidence threshold to be passed to the detector
        confThr=$OPTARG
        tstr=$(echo $confThr 10 | awk '{printf "%1.0f\n",$1*$2}')
        thrStr="dot""$tstr"
        confThrStr="$thrStr"
        #echo "tstr:$tstr, thrStr:$thrStr, confThr: $confThr, confThrStr: $confThrStr"
        ;;

    w)  # log file name to use
        lfn=$OPTARG
        touch $lfn
        echo "Log File is : $lfn"
        ;;
        
    x)  # Specify imageDirPrefixArg to pass to createList.py
        imageDirPrefixArg="--imageDirPrefix $OPTARG"
        ;;

    y)  # max IoU threshold to use for mAP and mAR
        maxIouThrArg="--maxIouThr $OPTARG"
        ;;

    z)  # create test video with ffmpeg
        createVideo=$OPTARG
        if [ $createVideo -ge '1' ] ; then
            createVideo=1
        else
            createVideo=0
        fi
        ;;

    *)  # unsupported argument
        echo "# ==================================================================================="
        echo "# Ignoring unknown or un-supported input argument: $OPTARG"
        echo "#    Supported input arguments are as below"
        echo "#         SHOULD be in the FORMAT/Value_Range as noted below. "
        echo "#         When NOT present, default values will be used as applicable"
        echo "# "
        echo "#       -a: Generate mAP Prime: calculate mAP based on annotation sizes as seen be CNN "
        echo "#               [0|resize_dim] : 0: disable. Else, input used as CNN input size. defautls to 512x512"
        echo "#                          NOTE: resize_dim value is used for both width & height"
        echo "#                          e.g., -a 300, would caculate the results with 300x300 size"
        echo "# "

        echo "# "
        echo "#       -c : Category-ID:  "
        echo "#               [0..90], where '0' is ALL and 'x' is mapped coco ID. "
        echo "#               This will be used to gather/calcuate the test results. "
        echo "# "
        echo "#       -d : test_data_set_directory:  "
        echo "#               relative path from ML_TEST_DATA_SET_ROOT "
        echo "#               e.g., "mytestdataset" should be at "$ML_TEST_DATA_SET_ROOT/mytestdataset"  "
        echo "# "
        echo "#       -f: Absolute path to detection output txt file from a CNN under test."
        echo "#            e.g., /path/to/cnn/model/under/test/foo_model_detection_output_results.txt"
        echo "#            The format of the output should be (each line): "
        echo "#            absolute-path-of-image-filename categoryId score xmin ymin xmax ymax time1 time2"
        echo "#            format: string int float float float float float int int"
        echo "#            where, time1&2 is time n micro-secs taken by CNN->forward and 'detections' "
        echo "#            processing respectively. If these are not available, set it to 0 "
        echo "# "
        echo "#       -h : h/modify detection output image_name format in the detections_output "
        echo "#                 file (provided using option -f above) "
        echo "#               The first element in each is the absolute path to the image, where "
        echo "#                   Image ID is part of the filename. ImageId is assigned based on index in "
        echo "#                   the image_set.json file."
        echo "#               e.g., 0: do not change. 1: change as required (default)"
        echo "# "
        echo "#       -j : images.json File_Name "
        echo "#               images file. Must be in "$mytestdataset" dir provided with '-d' option above "
        echo "#"
        echo "#       -l : labels.json file to be used when generating the GT. Absoulte path  "
        echo "#               e.g., input /mnt/data/models/mymodel/detectionOutput.txt "
        echo "#"
        echo "#       -m : labelmap.json file to be used when generating the GT. Absoulte path  "
        echo "#"
        echo "#       -p: plot ROC & PR Curves: "
        echo "#               [0/1/2/3], where  0: No Plot, 1: ROC Plot, 2: PR Plot, 3: both"
        echo "# "
        echo "#       -r: include all GT or remove images with no annotations "
        echo "#               "yes" : include all GT (default), "no" : remove images with no annotations"
        echo "# "
        echo "#       -s: Skip_GT:  "
        echo "#               [0/1] : 0 is don't skip generating GT and 1 is skip "
        echo "# "
        echo "#       -t: confidence_threshold:  "
        echo "#               [0.01 .. 0.90] : 0.01 is the default value"
        echo "# "
        echo "#       -w: log_file_name:  absolute path of the file to log into"
        echo "# "
        echo "#       -x: Image directory prefix to gather images from for GT"
        echo "#               Data or PreviewData (default)"
        echo "# "
        echo "#       -y: max IoU Thrshold to use for Precision and Recall metrics: "
        echo "#               [0.5 - 1.0, steps of 0.5, where 1.0 defaults to standard range 0.5 - 0.95"
        echo "# "
        echo "#       -z: create_video: "
        echo "#               [0/1], where  0: do not create video, 1: create video & populate detection output on test images"
        echo "# "
        echo "# =================================================================================== "
        ;;
 esac
done

# weed out errors
if [ ! -f "${detResultsFile}" ]; then
    echo "invalid argumet for option 'f'"
    echo " $detResultsFile :  No such file exists"
    echo "Unable to proceed further. Exiting the script"
    exit -1
else
    echo "Using detections_text file provided to evaluate your model under test $detectionsResultsInputFile"
fi

if [ ! -d "$ML_TEST_DATA_SET_ROOT/$dataSetDir" ]; then
    echo "invalid argumet for option 'd'"
    echo " $dataSetDir : No such directory exists"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi

if [ ! -e "$jsonLabelVectorFile" ] ; then
    echo "Cannot find labels file: $jsonLabelVectorFile"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi

if [ ! -e "$jsonLabelMapFile" ] ; then
    echo "Cannot find labelmap file: $jsonLabelMapFile"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi

##
# convert dataSetDir to string with no '/' so that we can use them for file_names
# using shell commands :-(
# The following strips the entire string preceding the last '/'
dsd="$dataRootDir""/""$dataSetDir"
dataSetDir2str=${dsd##*/}
echo " Converted $dataSetDir to $dataSetDir2str for use in Filenames"
##


## We are all set.
# setup labelvector file
labelVectorFile="$jsonLabelVectorFile"
# setup labelmap file
labelMapFile="$jsonLabelMapFile"

# setup image json file based on provided data src directory
imageJsonFileFullPath="$dataRootDir""/""$dataSetDir""/""$imageJsonFileName"
annoJsonFileFullPath="$dataRootDir""/""$dataSetDir""/""$annoJsonFileName"

# change to your needs : Log File name & Location
touch $lfn
echo "======================== START ======================" >> $lfn
date >> $lfn

#
# check and fail on environment variables here itself
# get the required input arguments, so that we can call other scripts.
#

testDir=$1
if [ "$testDir" ] || [ -z "$testDir" ] ; then
    echo " "
    echo "Testing with the following files, variables and values"
    echo "dataSetDir                                   : $ML_TEST_DATA_SET_ROOT/$dataSetDir "
    echo "anno json file                               : $annoJsonFileFullPath "
    echo "CNN Detection Results (Input) File           : $detResultsFile "
    echo "Modify Image Name in the above file          : $modifyImageNameInEachDetection "
    echo "categoryId_to_check_results (0: all)         : $catId"
    echo "skipCreateGT (1:yes, 0: No)                  : $skipCreateGT"
    echo "confidence_threshold [0.01 to 0.9]           : $confThr"
    echo "ap_ar (1:AP, 0: AR, -1: ALL)                 : $ap_ar"
    echo "area (small, medium, large, all)             : $area"
    echo "iouThr (0: ALL, 0.5, 0.75)                   : $iouThr"
    echo "labelVectorFile                              : $labelVectorFile"
    echo "labelMapFile                                 : $labelMapFile"
    echo "logFile (tail -f logFile for progress)       : $lfn"
    echo "plotRoC curve (1: yes, 0: no)                : $plotRoC"
    echo "plotPR  curve (1: yes, 0: no)                : $plotPR"
    echo "save o/p Images & create video(1:yes, 0:no)  : $createVideo"
    echo "Generate mAP Prime                           : $generate_mAP_prime"
    echo "imageDirPrefixArg                            : $imageDirPrefixArg"
    echo "IncludeAllGTArg                              : $includeAllGtArg"
    echo "max IoU Thr to use for mAP/mAR               : $maxIouThrArg"
    echo " "
    echo "NOTE: {CAFFE,COCO,ML_TEST_DATA_SET}_ROOT & PYTHONPATH must be set properly"
fi


echo "data root: $dataRootDir" 2>&1 | tee >> $lfn
echo "Caffe Root: $caffeRoot"  2>&1 | tee >> $lfn
echo "Coco Root: $cocoRoot"  2>&1 | tee >> $lfn
echo "log File: $lfn"  2>&1 | tee >> $lfn
echo "labelvector file: $labelVectorFile" 2>&1 | tee >> $lfn
echo "labelmap file: $labelMapFile" 2>&1 | tee >> $lfn
echo "dataSetDir: $ML_TEST_DATA_SET_ROOT/$dataSetDir " 2>&1 | tee >> $lfn
echo "CNN-Detections-File: $detResultsFile " 2>&1 | tee >> $lfn
echo "annotations json file: $annoJsonFileFullPath" 2>&1 | tee >> $lfn

cd $caffeRoot

## GT Full path file name
gtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""GT.json"
echo " GT File Full path: $gtFileName"
pgtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""primed_""GT.json"
echo " Primed_GT File Full path: $pgtFileName"

if [[ ! -z $labelRemap ]] ; then
  labelRemapArg="--remap_labels $labelRemap"
fi

echo " == 1 ================================================================== " 2>&1 | tee >> $lfn

#
# The Following needs to be called in sequence
#

myDir=$caffeRoot/scripts/data_prep
cd $myDir

if [ "$skipCreateGT" != 1 ]; then

    #echo "python createList.py $dataSetDir $labelVectorFile $imageJsonFileFullPath $imageDirPrefixArg $includeAllGtArg" 2>&1 | tee >> $lfn
    #python createList.py $dataSetDir $labelVectorFile $imageJsonFileFullPath $imageDirPrefixArg $includeAllGtArg 2>&1 | tee >> $lfn
    echo "python createList.py $dataSetDir $labelMapFile $labelVectorFile $annoJsonFileFullPath $imageDirPrefixArg $includeAllGtArg $labelRemapArg" 2>&1 | tee >> $lfn
    python createList.py $dataSetDir $labelMapFile $labelVectorFile $annoJsonFileFullPath $imageDirPrefixArg $includeAllGtArg $labelRemapArg 2>&1 | tee >> $lfn

    echo " == 2 ============================================================== " 2>&1 | tee >> $lfn

    echo "python Create_GT_json.py $dataSetDir $dataSetType $labelVectorFile $gtFileName " 2>&1 | tee >> $lfn
    python Create_GT_json.py $dataSetDir $dataSetType $labelVectorFile $gtFileName 2>&1 | tee >> $lfn

    echo " == 2A ============================================================== " 2>&1 | tee >> $lfn

    # Note: getImageSize.sh uses the same imageList.txt file as ssd-detect-FLIR binary
    # changes are in get_image_size.py script, to strip the extension before indexing
    echo "bash getImageSize.sh $dataSetDir $dataSetType $gtFileName" 2>&1 | tee >> $lfn
    bash getImageSize.sh $dataSetDir $dataSetType $gtFileName 2>&1 | tee >> $lfn

    echo " == 2B ============================================================== " 2>&1 | tee >> $lfn
fi

echo " == 3 ============================================================== " 2>&1 | tee >> $lfn

#
# let us call the convert script if requried to convert the image name element in each detection
#       in the detectionOutput file (input as option -f), to our needs i.e., absolute path and
#       imageId appended to filename
#

detResultsOutFile="${detResultsFile/.txt/.log}"
detResultsInFile="${detResultsFile}"

if [ "$modifyImageNameInEachDetection" -eq 1 ]; then

    echo " == 3A ==================detection results IN file $detResultsInFile ================== " 2>&1 | tee >> $lfn

    pyScript2call="rename_detection_log_filenames.py"
    pathPrefix="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""Data"
    #echo "python $pyScript2call $imageJsonFileFullPath $detResultsFile $pathPrefix " 2>&1 | tee >> $lfn
    #python $pyScript2call $imageJsonFileFullPath $detResultsFile $pathPrefix 2>&1 | tee >> $lfn
    echo "python $pyScript2call $annoJsonFileFullPath $detResultsFile $pathPrefix " 2>&1 | tee >> $lfn
    python $pyScript2call $annoJsonFileFullPath $detResultsFile $pathPrefix 2>&1 | tee >> $lfn

    echo " == 3B ===================================================== " 2>&1 | tee >> $lfn

    #setup detResultsInFile accordingly
    detResultsInFile="${detResultsOutFile}"
fi

echo " == 4 ============================================================== " 2>&1 | tee >> $lfn
echo " == 4A ==================detection results IN file $detResultsInFile ================== " 2>&1 | tee >> $lfn

cd $caffeRoot
# convert the .txt detection results input file into json file. This script generates $detResultsFile.json
scriptDir=$caffeRoot/scripts/data_prep
#echo "python $scriptDir/txt2json.py $detResultsFile" 2>&1 | tee >> $lfn
#python $scriptDir/txt2json.py $detResultsFile 2>&1 | tee >> $lfn
echo "python $scriptDir/txt2json.py $detResultsInFile" 2>&1 | tee >> $lfn
python $scriptDir/txt2json.py $detResultsInFile 2>&1 | tee >> $lfn

# let us always evaluate with the requested confThr
echo " == 5 ========== Evaluation with confThr : $confThr ($thrStr) ================ " 2>&1 | tee >> $lfn

cd $cocoRoot

# setup detection results json file to pass it for evaluation
detResultsJsonFile="${detResultsFile/.txt/.json}"    # ${my_file/.extold/.extnew}

echo "detResultsJsonFile: $detResultsJsonFile" 2>&1 | tee >> $lfn

## Following two are just dummy variables
viewImages=0            # dummy to pass to generic script below
numberOfImagesToView=0  # dummy to pass to generic script below
resize_height=0
resize_width=0
resultsOutputDir=$cnnEvalResultsDir

## 
#   Generate regular evaluation results only when prime_mAP is not required
##
if [ $generate_mAP_prime -eq 0 ]; then

    echo " == 5A ===================== regular mAP ========================== " 2>&1 | tee >> $lfn

    # setup directory to save images with detections annotated
    detectionImagesOutputDir="None"
    if [ "${createVideo}" -eq 1 ] ; then
        detectionImagesOutputDir="$cnnEvalResultsDir""/workspace/detections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
    fi

    echo "python PythonAPI/pycocoEvalDemo.py $gtFileName $dataSetDir $resultsOutputDir $detResultsJsonFile $detectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg" 2>&1 | tee >> $lfn
    python PythonAPI/pycocoEvalDemo.py $gtFileName $dataSetDir $resultsOutputDir $detResultsJsonFile $detectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg 2>&1 | tee -a >> $lfn 

fi

# ========================================================
#       generate detection and GT files with resized
#       bboxes and calc mAP if requested otherwise
# ========================================================

resizeBboxScript=$caffeRoot/scripts/data_prep/resizeBbox.py
if [ $generate_mAP_prime -ne 0 ]; then

    echo " == 5B ========== Generating mAP Prime ============================ " 2>&1 | tee >> $lfn

    resize_height=$generate_mAP_prime
    resize_width=$generate_mAP_prime

    pdetectionImagesOutputDir="None"
    if [ "${createVideo}" -eq 1 ] ; then
        #pdetectionImagesOutputDir="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""pdetections"
        pdetectionImagesOutputDir="$cnnEvalResultsDir""/workspace/""pdetections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
    fi

    #pdetResultsJsonFile="$resultsOutputDir""/""primed_detectionResults_catId_""$catId""_""$thrStr"".json"
    pdetResultsJsonFile="$cnnEvalResultsDir""/workspace/""primed_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"

    echo "creating Primed GT Annotations & detections based on resized image into CNN" 2>&1 | tee >> $lfn
    echo "python $resizeBboxScript $gtFileName $pgtFileName $detResultsJsonFile $pdetResultsJsonFile $resize_height $resize_width" 2>&1 | tee >> $lfn
    python $resizeBboxScript $gtFileName $pgtFileName $detResultsJsonFile $pdetResultsJsonFile $resize_height $resize_width 2>&1 | tee >> $lfn

    echo "python PythonAPI/pycocoEvalDemo.py $pgtFileName $dataSetDir $resultsOutputDir $pdetResultsJsonFile $pdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg" 2>&1 | tee >> $lfn
    python PythonAPI/pycocoEvalDemo.py $pgtFileName $dataSetDir $resultsOutputDir $pdetResultsJsonFile $pdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg 2>&1 | tee >> $lfn

fi

echo " ==8============================================================== " 2>&1 | tee >> $lfn

## call RoC / P-R scripts to plot if requested 
## Only do these if catId is > 0. 
## NOTE: coco style PR curves will be generated when catId is 0
if [ "${plotRoC}" -eq 1 ] || [ "${plotPR}" -eq 1 ] && [ "${catId}" -ne 0 ]; then

    myDir=$caffeRoot/scripts/data_prep
    cd $myDir

    echo "=========== PLOT RoC: $plotRoC P-R: $plotPR ====================" 2>&1 | tee >> $lfn
    echo "=========== PLOT RoC: $plotRoC P-R: $plotPR ====================" #2>&1 | tee >> $lfn

    detResultsJsonFile="$cnnEvalResultsDir""/workspace/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"
    detResultsJsonFilePrefix="$cnnEvalResultsDir""/workspace/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId"
    datasetDir2Plot="$dataRootDir""/""$dataSetDir""/""$dataTypeStr"

    if [ "${plotRoC}" -eq 1 ] ; then
        ## call the plotting script(s) with dataSrcDir, gtFile, detResultsFile, catId,
        echo "====== TODO : FIX PlotROC plots code too =========" 2>&1 | tee >> $lfn
        echo "====== TODO : FIX PlotROC plots code too =========" 
        #echo "python plotROC.py $datasetDir2Plot $gtFileName $detResultsJsonFile $catId" 2>&1 | tee >> $lfn
        #python plotROC.py $datasetDir2Plot $gtFileName $detResultsJsonFile $catId 2>&1 | tee >> $lfn
    fi

    if [ "${plotPR}" -eq 1 ] ; then
        echo "python plotPR.py $datasetDir2Plot $gtFileName $detResultsJsonFile $catId" 2>&1 | tee >> $lfn
        python plotPR.py $datasetDir2Plot $gtFileName $detResultsJsonFile $catId 2>&1 | tee >> $lfn
    fi

fi

echo " ==9============================================================== " 2>&1 | tee >> $lfn

if [ "${createVideo}" -eq 1 ] ; then
    if ! [ -x "$(command -v ffmpeg)" ]; then
        echo 'Error: ffmpeg is not installed.' 2>&1 | tee >> $lfn
        exit -1

    else
    
        if [ $generate_mAP_prime -eq 0 ]; then
            testVideoName="$cnnEvalResultsDir""/workspace/""video_""$dataSetDir2str""_catId_""$catId""_""$thrStr"".mp4"
            ffmpeg -y -framerate 10 -pattern_type glob -i "$detectionImagesOutputDir""/""*.jpeg" $testVideoName 2>&1 | tee >> /dev/null
            echo "Test video available at $testVideoName" 2>&1 | tee >> $lfn

        else

            ## generate for prime case also
            ptestVideoName="$cnnEvalResultsDir""/workspace/""pvideo_""$dataSetDir2str""_catId_""$catId""_""$thrStr"".mp4"
            ffmpeg -y -framerate 10 -pattern_type glob -i "$pdetectionImagesOutputDir""/""*.jpeg" $ptestVideoName 2>&1 | tee >> /dev/null
            echo "Test video for prime-mAP available at $ptestVideoName" 2>&1 | tee >> $lfn
        fi
    fi
fi

echo " ==10============================================================== " 2>&1 | tee >> $lfn
echo " Timing info (if available in detection log)" 2>&1 | tee >> $lfn
LastRecord=$(tail -1 $detResultsFile | awk '{print $1}')
if [ $LastRecord == "LastRecord" ]; then
  DPS=$(tail -1 $detResultsFile | awk '{print $9}')
  FPS=$(tail -1 $detResultsFile | awk '{print $10}')
  echo "  Detections per Second (DPS): $DPS" 2>&1 | tee >> $lfn
  echo "  Frames per Second (FPS): $FPS" 2>&1 | tee >> $lfn
else
  echo "  Dps/Fps information not available in log file: $detResultsFile" 2>&1 | tee >> $lfn
fi
echo " ================================================================== " 2>&1 | tee >> $lfn


echo " DONE!! Check Results File: $tarFileName & Logfile $lfn" 2>&1 | tee >> $lfn

echo "======================== END ======================" >> $lfn

echo "  Logfile $lfn"
echo " DONE!! "

