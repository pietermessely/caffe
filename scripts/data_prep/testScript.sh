#!/bin/bash

##############################################################################
## Purpose
#
#       Validation Testing of FLIR Trained models, to help Team_DML
#         in better understanding the failures and further improve CNN under test.
#
#       At a glance, this script does the following:
#               Scoring per coco format
#               Scoring per Flir specific needs: FP, FN, Detections Missed.
#               Scoring on a specific category or on all available categories.
#               Calculates Detection/Sec and Frames_processed/Sec
#               Optionally, batch input images and check performance
#               Optionally, view detections on images
#               Logs CPU & GPU processor and memory resource usage
#               Generates results files and logs.
#               Creates one date_time stamped tgz file in the model_UT dir.
##############################################################################

set -e


dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
modelsRootDir="${FLIR_MODELS_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"
pyPath="${PYTHONPATH:-""}"

echo "ML_TEST_DATA_SET_ROOT: $ML_TEST_DATA_SET_ROOT"
echo "FLIR_MODELS_ROOT: $FLIR_MODELS_ROOT"
echo "CAFFE_ROOT: $CAFFE_ROOT"
echo "COCO_ROOT: $COCO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

if [ -z "$dataRootDir" ]; then
    echo "Need to setup ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi
if [ -z "$modelsRootDir" ]; then
    echo "Need to setup FLIR_MODELS_ROOT. Cannot Proceed Further"
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

if [ -z $(which pidstat) ]; then
    echo "Need pidstat. Cannot Proceed Further"
    echo "  echo sudo apt-get install sysstat"
    exit
fi

if [ -z $(which nvidia-smi) ]; then
    echo "Need nvidia-smi. Cannot Proceed Further"
    echo "  Did you install NVIDIA drivers version 384.111? This may change"
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
lfn="$logPrefixDir/k.log"

if [ ! -d "$logPrefixDir" ]; then
    echo "Creating $logPrefixDir" 2>&1
    mkdir $logPrefixDir 2>&1
fi

touch $lfn

# defaults for input arguments
dataSetDir="test_astoria-aerial"
modelFileSrcDir="object_detectors/coco/ssd_inception-v1_512x512"
modelFileName="inceptionV1_coco_SSD_512x512_iter_360000.caffemodel"

# CNN under test
cnnUnderTest="ssd_detect_FLIR"
dataSetType=1           # (1) Test dataset (0) train_val
catId=0                 # ALL classes (present in ground truth)

#Steps:
#  1. Convert all ground-truth to format that we need. ("CreateGT")
#  2. For each batch of images,
#     2a. Compute scoring metric ("RunDetEval")
#     2b. Record details related to scoring metric. (e.g., bounding box matches).
#  3. Generate diagnostic images (draw detections and bounding boxes). ("viewImages")

# Adust the following flags to skip some or all of the above steps.
# Common scenerios:
# -   End to end testing:  skipCreateGT=0, skipRunDetEval=0, viewImages=?
# -       Re-run testing:  skipCreateGT=1, skipRunDetEval=0, viewImages=?
# - Re-run, generate dbg:  skipCreateGT=1, skipRunDetEval=1, viewImages=1

skipCreateGT=0          # 1: skip conversion all all ground truth files
skipRunDetEval=0        # 1: skip evaluation tests - useful to check scores from last eval
viewImages=0            # yes/no view detection images(1) or not (0)
numberOfImagesToView=2  # The first N images of each category are viewed.

batch_size=13       # batch size for running detector on multiple images in GPU in parallel

# These flag controls which metrics are computed. There are lots of metrics, so sometimes
# it is not convenient (and time consuming) to compute of them. So this lets you only compute
# a subset of the metrics.
ap_ar=-1              # coco scores: -1 = All, 1 = AP, 0 = AR
#ap_ar=0              # coco scores: -1 = All, 1 = AP, 0 = AR
#ap_ar=1              # coco scores: -1 = All, 1 = AP, 0 = AR
iouThr=0          #0:all, 0.5, 0.75

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
confThrList=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
confThrStr=("dot3" "dot4" "dot5" "dot6" "dot7" "dot8" "dot9")
areaStr=("small" "medium" "large" "all")
area="all"          #all, small, medium, large

generate_mAP_prime=0

# By default disable creating test video. This will also disable saving output images
createVideo=0

# annotations file to use for creating the GT
annoJsonFileName="annotation_set.json"
annoJsonFileFullPath=""

# default image directory prefix to use to grab the images from for GT
imageDirPrefixArg="--imageDirPrefix PreviewData"

# include all GT or remove images with no annotations
includeAllGtArg="--includeAllGt yes"

labelRemapArg=""

# GPU to use to run the detector. Default is '0'
gpu_device_id=0

# mean value to be subtracted from the input image data. Default is "104,117,123"
mean_value="104,117,123" 

# scaling factor used to mutiply the mean-subtracted images. Default is "1.0"
normalize_value="1.0"

# ====================================================================
# ALL Defaults should have been defined above
#
# "change to your needs" identifies sections to modify per your needs
# ====================================================================


# ===================================================================================
# Read Input Arguments if any and update the variables
#
# All arguments are optional.
#       If present, SHOULD be in the FORMAT as noted below.
#       When NOT present, default values assigned above will be used
#
# Format of Input Args are:
#
#       -a: mAP Prime:
#               [0|resize] : 0 (default) disables calculating mAP-Prime. resize: enables this,
#                        using the value as the resize value to resize frame
#
#       -b: batch_size:
#               [1..512] : depending on your system, higher values may result in errors
#
#       -c : Category-ID:
#               [0..90], where '0' is ALL and 'x' is mapped coco ID.
#               This will be used to gather/calcuate the test results.
#
#       -d : test_data_set_directory relative to ML_TEST_DATA_SET_ROOT
#               e.g., input "mytestdataset" dir should be at "$ML_TEST_DATA_SET_ROOT/mytestdataset"
#
#       -e: Skip_Detection_Eval:
#               [0/1] : 0 is don't skip and 1 is skip
#
#       -f : Model_File_Name:
#               model file to be used. Expected to be in "$Model_File_Dir"
#
#       -g : GPU device ID to use:
#               device id of the GPU to use. default: '0'
#
#       -i: view_detection_results_on_images:
#               [0..9999] : 0 is no and +ve number is used as number of images to show.
#                Spawns images with GT & DT bboxes
#
#       -l: remap_labels.json
#               Json file specifying lables to remap in the ground truth.
#               Ex: { "src_label" : :"dst_label",  "potato" : "person" }
#
#       -m : Model_File_Dir relative to FLIR_MODELS_ROOT:
#               e.g., input "object_detectors/aerial/ssd_vgg_512x512" dir
#                       should be at
#               "$FLIR_MODELS_ROOT/object_detectors/aerial/ssd_vgg_512x512"
#
#       -n: normalize_value:
#               "1.0" (default) : Specifying the scaling factor used to mutiply the mean-subtracted images
#
#       -p: plot ROC & PR Curves:
#               [0/1/2/3], where  0: No Plot, 1: ROC Plot, 2: PR Plot, 3: both
#
#       -r: include all GT or remove images with no annotations
#               "yes" : include all GT (default), "no" : remove images with no annotations
#
#       -s: Skip_GT:
#               [0/1] : 0 is don't skip and 1 is skip
#
#       -t: confidence_threshold:
#               [0.01 .. 0.90] : 0.01 is the default, to compute metrics in the entire confidence threshold range.
#                                Note, time to compute metrics will be longer with default value
#       -u: mean_value:
#               "104,117,123" (default) : Specifying the mean value to be subtracted from the input image data
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
#               TODO: Output Images are saved in test dataset directory under
#                       'Detections' & 'pdetections' folder for the corresponding
#                       mAP or mAP-Prime by default. Need to do this only with video creation
#                       option is set and clean up existing directories before saving.
#
#       e.g., ./testScript.sh -m "object_detectors/aerial/ssd_vgg_512x512" \
#                                -f "aerial_ssd_vgg_512x512_iter_11000.caffemodel" \
#                                -d "mytestdataset"
#
# ===================================================================================

while getopts a:b:c:d:e:f:g:i:l:m:n:p:r:s:t:u:w:x:y:z: option
do
 case "${option}" in

    a) # generate map-prime also
        ##
        # NOTE - This value will be used as height AND width to 'resize' the image
        ##
        generate_mAP_prime=$OPTARG

        #if [ $generate_mAP_prime -ge '1' ] ; then
        #    generate_mAP_prime=1
        #else
        #    generate_mAP_prime=0
        #fi

        echo "Generate mAP Prime: $generate_mAP_prime"
        ;;

    b)  # batch_size
        batch_size=${OPTARG}
        if [ "$batch_size" -le '0' ] || [ "$batch_size" -gt '512' ]; then
            batch_size=1
            echo "changed batch_size to: $batch_size"
        fi
        ;;

    c)  # category ID
        catId=${OPTARG}
        ;;

    d)  # dataset directory to be used
        dataSetDir=${OPTARG}
        ;;

    e)  # skit detection evaluation testing of the model-under-test
        skipRunDetEval=$OPTARG
        #echo "skipRunDetEval : $skipRunDetEval"
        if [ $skipRunDetEval -lt '0' ] || [ $skipRunDetEval -gt '1' ]; then
            skipRunDetEval=0
            echo "changed skipRunDetEval to: $skipRunDetEval"
        fi
        ;;


    f)  # file name of the model to be tested
        modelFileName=$OPTARG
        ;;

    g)  # gpu_device_id to use
        gpu_device_id=${OPTARG}
        echo "gpu_device_id being used is: $gpu_device_id"
        ;;

    i)  # number of images to view detection results on
        viewImages=$OPTARG
        if [ $viewImages -lt '0' ] || [ $viewImages -gt '999' ]; then
            viewImages=0
            echo "changed number of Images to view to: $viewImages"
        fi
        numberOfImagesToView=${viewImages}

        # setup viewImages as boolean value
        if [ $viewImages -gt '0' ]; then
            viewImages=1
        fi

        ;;

    l)  # label remapping json file
        labelRemap=$OPTARG
        ;;

    m)  # source directory of the model file under test
        modelFileSrcDir=$OPTARG
        ;;

    n)  # scaling factor used to mutiply the mean-subtracted images
        normalize_value=${OPTARG}
        echo "normalize_value being used is: $normalize_value"
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

    s)  # skit creating GT from the data set being used for testing
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
        ;;

    u)  # mean value to be subtracted from the input image data
        mean_value=${OPTARG}
        echo "mean_value being used is: $mean_value"
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
        else :
            createVideo=0
        fi
        ;;

    *)  # unsupported argument
        echo "# ==================================================================================="
        echo "# Ignoring unknown or un-supported input argument: $OPTARG"
        echo "#    Supported input arguments are as below, where ALL are optional"
        echo "#         When present, SHOULD be in the FORMAT/Value_Range as noted below. "
        echo "#         When NOT present, default values will be used "
        echo "# "
        echo "#       -a: Generate mAP Prime: calculate mAP based on annotation sizes as seen be CNN "
        echo "#               [0|resize_dim] : 0: disable. Else, input used as CNN input size. defautls to 512x512"
        echo "#                          NOTE: resize_dim value is used for both width & height"
        echo "#                          e.g., -a 300, would caculate the results with 300x300 size"
        echo "# "
        echo "#       -b: batch_size:  "
        echo "#               [1..512] : depending on your system, higher values may result in errors "
        echo "# "
        echo "#       -c : Category-ID:  "
        echo "#               [0..90], where '0' is ALL and 'x' is mapped coco ID. "
        echo "#               This will be used to gather/calcuate the test results. "
        echo "# "
        echo "#       -d : test_data_set_directory:  "
        echo "#               relative path from ML_TEST_DATA_SET_ROOT "
        echo "#               e.g., input ""mytestdataset"" dir should be at ""$ML_TEST_DATA_SET_ROOT/mytestdataset""       "
        echo "# "
        echo "#       -e: Skip_Detection_Eval: "
        echo "#               [0/1] : 0 is don't skip and 1 is skip"
        echo "# "
        echo "#       -f : Model_File_Name:  "
        echo "#               model file to be used. Expected to be in model_file_dir (option -m below) "
        echo "# "
        echo "#       -g: GPU device id to use: "
        echo "#               [0..x] : 0 is default. x=(num_of_gpus_in_your_system - 1)"
        echo "# "
        echo "#       -i: view_detection_results_on_images: "
        echo "#               [0..9999] : 0 is no and +ve number is used as number of images to show. "
        echo "#                Spawns images with GT & DT bboxes "
        echo "# "
        echo "#       -l : label_remap.json File_Name "
        echo "#               .json file. Provide full path."
        echo "#"
        echo "#       -m : Model_File_Dir:  "
        echo "#               relative path from FLIR_MODELS_ROOT "
        echo "#               e.g., input ""object_detectors/aerial/ssd_vgg_512x512"" dir  "
        echo "#                       should be at   "
        echo "#               ""$FLIR_MODELS_ROOT/object_detectors/aerial/ssd_vgg_512x512""       "
        echo "# "
        echo "#       -n: normalize_value:  "
        echo "#               ""1.0"" (default) : Specifying the scaling factor used to mutiply the mean-subtracted images"
        echo "# "
        echo "#       -p: plot ROC & PR Curves: "
        echo "#               [0/1/2/3], where  0: No Plot, 1: ROC Plot, 2: PR Plot, 3: both"
        echo "# "
        echo "#       -r: include all GT or remove images with no annotations "
        echo "#               ""yes"" : include all GT (default), ""no"" : remove images with no annotations"
        echo "#"
        echo "#       -s: Skip_GT:  "
        echo "#               [0/1] : 0 is don't skip and 1 is skip "
        echo "# "
        echo "#       -t: confidence_threshold:  "
        echo "#               [0.01 .. 0.90] : 0.01 is the default value"
        echo "# "
        echo "#       -u: mean_value:  " 
        echo "#               ""104,117,123"" (default) : Specifying the mean value to be subtracted from the input image data"
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
        echo "#               [0/1], where  0: not create video, 1: create video"
        echo "# "
        echo "# =================================================================================== "
        ;;
 esac
done

# weed out errors
if [ ! -d "$FLIR_MODELS_ROOT/$modelFileSrcDir" ]; then
    echo "invalid argumet for option 'm'"
    echo " $modelFileSrcDir : No such directory exists"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi

if [ ! -f "$FLIR_MODELS_ROOT/$modelFileSrcDir/$modelFileName" ] ; then
    echo "invalid argumet for option 'f'"
    echo " $modelFileName :  No such file exists"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi


if [ ! -d "$ML_TEST_DATA_SET_ROOT/$dataSetDir" ]; then
    echo "invalid argumet for option 'd'"
    echo " $dataSetDir : No such directory exists"
    echo "Unable to proceed further. Exiting the script"
    exit -1
fi

##
# convert dataSetDir and modelFileSrcDir to string with no '/' so that we can use them for file_names
# using shell commands :-(
# The following strips the entire string preceding the last '/'
dataSetDir2str=${dataSetDir##*/}
echo " Converted $dataSetDir to $dataSetDir2str for use in Filenames"
modelFileSrcDir2str=${modelFileSrcDir##*/}
echo " Converted $modelFileSrcDir to $modelFileSrcDir2str for use in Filenames"
##

#
##For DEBUG Use: echo the arguments to be used for these tests
#
#echo " batch_size: $batch_size"
#echo " catId: $catId "
#echo " confidence Threshold: $confThr "
#echo " data set src dir: $dataSetDir "
#echo " Model file src dir: $modelFileSrcDir "
#echo " Model File under test: $modelFileName "
#echo " Skip creating GT from Data set: $skipCreateGT "
#echo " Show detection Results on images? : $viewImages "
#echo " Number of images to view detection, FN & FP results on: $numberOfImagesToView "

## We are all set.


# setup dataSet Type String based on data type
if [ "$dataSetType" == 1 ]; then
    dataTypeStr="test"
fi

if [ "$dataSetType" != 1 ]; then
    echo "dataSet Type $dataSetType NOT supported yet." 2>&1 | tee >> $lfn
    exit
fi

# setup labelmap file based on provided models File src directory
labelMapFile="$modelsRootDir""/""$modelFileSrcDir""/""labelmap.json"
labelVectorFile="$modelsRootDir""/""$modelFileSrcDir""/""labelvector.json"

if [ ! -f "$labelMapFile" ]; then
    echo "labelmap.json file not found!"
fi

if [ ! -f "$labelVectorFile" ]; then
    echo "labelvector.json file not found!"
fi

# setup image json file based on provided data src directory
annoJsonFileFullPath="$dataRootDir""/""$dataSetDir""/""$annoJsonFileName"

echo "======================== START ======================" >> $lfn
date >> $lfn

#
# check and fail on environment variables here itself
# get the required input arguments, so that we can call other scripts.
#

testDir=$1
if [ "$testDir" ] || [ -z "$testDir" ] ; then
    echo "Testing with the following files, variables and values"
    echo "dataSetDir                                   : $ML_TEST_DATA_SET_ROOT/$dataSetDir "
    echo "annotations json file                        : $annoJsonFileFullPath "
    echo "dataSetType(1: test, 0: train_val)           : $dataSetType"
    echo "modelFileSrcDir                              : $FLIR_MODELS_ROOT/$modelFileSrcDir "
    echo "modelFileName                                : $modelFileSrcDir/$modelFileName "
    echo "cnnUnderTest                                 : $cnnUnderTest "
    echo "categoryId_to_check_results (0: all)         : $catId"
    echo "skipCreateGT (1:yes, 0: No)                  : $skipCreateGT"
    echo "skipRunDetEval (1:yes, 0: No)                : $skipRunDetEval"
    echo "viewImages(1:yes, 0: No)                     : $viewImages"
    echo "numberOfImagesToView                         : $numberOfImagesToView"
    echo "confidence_threshold [0.01 to 0.9]           : $confThr"
    echo "batch_size(1..512)                           : $batch_size"
    echo "ap_ar (1:AP, 0: AR, -1: ALL)                 : $ap_ar"
    echo "area (small, medium, large, all)             : $area"
    echo "iouThr (0: ALL, 0.5, 0.75)                   : $iouThr"
    echo "labelMapFile                                 : $labelMapFile"
    echo "logFile (tail -f logFile for progress)       : $lfn"
    echo "plotRoC curve (1: yes, 0: no)                : $plotRoC"
    echo "plotPR  curve (1: yes, 0: no)                : $plotPR"
    echo "save o/p Images & create video(1:yes, 0:no)  : $createVideo"
    echo "IncludeAllGTArg                              : $includeAllGtArg"
    echo "ImagePrefixDirArg                            : $imageDirPrefixArg"
    echo "max IoU Thr to use for mAP/mAR               : $maxIouThrArg"
    echo "GPU device id to use                         : $gpu_device_id"
    echo "mean_value to use                            : $mean_value"
    echo "normalize_value to use                       : $normalize_value"
    echo " "
    echo "NOTE: {CAFFE,COCO,ML_TEST_DATA_SET,FLIR_MODELS}_ROOT & PYTHONPATH must be set properly"
fi


echo "data root: $dataRootDir" 2>&1 | tee >> $lfn
echo "models root: $modelsRootDir" 2>&1 | tee >> $lfn
echo "Caffe Root: $caffeRoot"  2>&1 | tee >> $lfn
echo "Coco Root: $cocoRoot"  2>&1 | tee >> $lfn
echo "log File: $lfn"  2>&1 | tee >> $lfn
echo "labelmap file: $labelMapFile" 2>&1 | tee >> $lfn
echo "dataSetDir: $ML_TEST_DATA_SET_ROOT/$dataSetDir " 2>&1 | tee >> $lfn
echo "dataSetType(1: test, 0: train_val): $dataSetType" 2>&1 | tee >> $lfn
echo "modelFileSrcDir: $FLIR_MODELS_ROOT/$modelFileSrcDir " 2>&1 | tee >> $lfn
echo "modelFileName: $modelFileSrcDir/$modelFileName " 2>&1 | tee >> $lfn
echo "annotations json file: $annoJsonFileFullPath" 2>&1 | tee >> $lfn

# TODO Add the following as input arguments to this script
#echo "$cnnUnderTest" ... need this eventually when testing non-ssd types?

cd $caffeRoot

## GT Full path file name
## setup GT file names
gtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""GT.json"
echo " GT File Full path: $gtFileName"
pgtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""primed_""GT.json"
echo " Primed_GT File Full path: $pgtFileName"

if [[ ! -z $labelRemap ]] ; then
  labelRemapArg="--remap_labels $labelRemap"
fi

echo " ==0=================================================================== " 2>&1 | tee >> $lfn

#
# The Following needs to be called in sequence
#

myDir=$caffeRoot/scripts/data_prep
cd $myDir

if [ "$skipCreateGT" != 1 ]; then
    echo "python createList.py $dataSetDir $labelMapFile $labelVectorFile $annoJsonFileFullPath $imageDirPrefixArg $includeAllGtArg $labelRemapArg" 2>&1 | tee >> $lfn
    python createList.py $dataSetDir $labelMapFile $labelVectorFile $annoJsonFileFullPath $imageDirPrefixArg $includeAllGtArg $labelRemapArg 2>&1 | tee >> $lfn
fi

echo " ==1=================================================================== " 2>&1 | tee >> $lfn

if [ "$dataSetType" != 1 ]; then
    echo "./create_data.sh $dataSetDir" 2>&1 | tee >> $lfn
    ./create_data.sh $dataSetDir 2>&1 | tee >> $lfn
fi

echo " ==2============================================================== " 2>&1 | tee >> $lfn

if [ "$skipCreateGT" != 1 ]; then
    echo "python Create_GT_json.py $dataSetDir $dataSetType $labelVectorFile $gtFileName" 2>&1 | tee >> $lfn
    python Create_GT_json.py $dataSetDir $dataSetType $labelVectorFile $gtFileName 2>&1 | tee >> $lfn
fi

echo " ==3============================================================== " 2>&1 | tee >> $lfn


# Note: getImageSize.sh uses the same imageList.txt file as ssd-detect-FLIR binary
# changes are in get_image_size.py script, to strip the extension before indexing
if [ "$skipCreateGT" != 1 ]; then
    echo "bash getImageSize.sh $dataSetDir $dataSetType $gtFileName " 2>&1 | tee >> $lfn
    bash getImageSize.sh $dataSetDir $dataSetType $gtFileName 2>&1 | tee >> $lfn
fi

echo " ==4============================================================== " 2>&1 | tee >> $lfn

cd $caffeRoot

# put intermediate results in 'workspace' subfolder
workspaceDir="$modelsRootDir""/""$modelFileSrcDir""/""workspace"
if [ ! -d "$workspaceDir" ]; then
    mkdir $workspaceDir
fi

# get iteration mumber from the model file or the string after last '_' from the model file name
mfname="${modelFileName/.*/}"
iterNum=${mfname##*_}
echo "iterNum: $iterNum" 2>&1 | tee >> $lfn

modelFile2Test="$modelsRootDir""/""$modelFileSrcDir""/""$modelFileName"
#detResultsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr""_iter_""$iterNum"".txt"
detResultsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".txt"
deployProtoFile="$modelsRootDir""/""$modelFileSrcDir""/""deploy.prototxt"
imageListFile="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""$dataTypeStr""ImageList.txt"

FLAGS="-confidence_threshold=""$confThr"" -view_image=0 -out_file=""$detResultsFile -batch_size=$batch_size -gpu_id=$gpu_device_id"

# change to your needs : GPU Stats
gpuStatsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""gpuStats_catId_""$catId"".csv"
# 10, 1 msec is useful to look at the utilization gaps much more finely. 100ms is default
gpuStatsTimerInMsecs=100
gpuStatsScript="$caffeRoot""/scripts/data_prep/gpuStatsCollector.sh"

# change to your needs : CPU Stats
cpuStatsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""cpuStats_catId_""$catId"".log"
# number of snapshots to take @ 1 sec interval (minimum snapshot interval)
cpuStatsSnapshotCounter=300
cpuStatsScript="$caffeRoot""/scripts/data_prep/cpuStatsCollector.sh"

myDir=$caffeRoot/build/examples/ssd

if [ "$skipRunDetEval" != 1 ]; then

    # change to your needs : GPU Stats
    # Need to kick start the GPU stats collection scripts & run in background
    echo "bash $gpuStatsScript $gpuStatsFile $gpuStatsTimerInMsecs&"
    bash $gpuStatsScript $gpuStatsFile $gpuStatsTimerInMsecs&
    gpuPid=$!
    echo "gpuStatsScript PID is: ${gpuPid} "

    # change to your needs : CPU Stats
    # Need to kick start the CPU stats collection scripts & run in background
    echo "bash $cpuStatsScript $cpuStatsFile $cpuStatsSnapshotCounter&"
    bash $cpuStatsScript $cpuStatsFile $cpuStatsSnapshotCounter&
    cpuPid=$!
    echo "cpuStatsScript PID is: ${cpuPid} "

    echo "deploy.proto: $deployProtoFile " 2>&1 | tee >> $lfn
    echo "modelFile: $modelFileName " 2>&1 | tee >> $lfn
    echo "modelFile2Test: $modelFile2Test " 2>&1 | tee >> $lfn
    echo "imageList: $imageListFile" 2>&1 | tee >> $lfn

    # run the network under test 
    #detResultsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr""_iter_""$iterNum"".txt"
    detResultsFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".txt"
    FLAGS="-confidence_threshold=""$confThr"" -view_image=0 -out_file=""$detResultsFile -batch_size=$batch_size -gpu_id=$gpu_device_id -mean_value=""$mean_value"" -normalize_value=""$normalize_value"
    echo "$myDir/$cnnUnderTest $deployProtoFile $modelFile2Test $imageListFile $FLAGS" 2>&1 | tee >> $lfn
    $myDir/$cnnUnderTest $deployProtoFile $modelFile2Test $imageListFile $FLAGS 2>&1 | tee >> $lfn

    echo " == 5 default ==== $thrStr ====== $confThr ====================================================== " 2>&1 | tee >> $lfn

    # change to your needs : append detection results to log file also
    # cat the det results file into log file
    #cat $detResultsFile 2>&1 | tee >> $lfn
    tail -5 $detResultsFile 2>&1 | tee >> $lfn

    # convert the .txt detection results file into json file. This script generates $detResultsFile.json
    scriptDir=$caffeRoot/scripts/data_prep
    echo "python $scriptDir/txt2json.py $detResultsFile" 2>&1 | tee >> $lfn
    python $scriptDir/txt2json.py $detResultsFile 2>&1 | tee >> $lfn

    # change to your needs : GPU Stats
    # need to stop the gpustats collector.
    kill -9 ${gpuPid} 2>&1 | tee >> $lfn
    killall -9 nvidia-smi 2>&1 | tee >> $lfn

    # change to your needs : CPU Stats
    # need to stop the cpustats collector.
    kill -9 ${cpuPid} 2>&1 | tee >> $lfn
    killall -9 pidstat 2>&1 | tee >> $lfn

    ## get the top 4 usage entries for both the GPU and CPU resources
    echo ======== GPU Resources ========== 2>&1 | tee >> $lfn
    echo ===== Processor % ========== 2>&1 | tee >> $lfn
    awk '{ print $1 }' $gpuStatsFile | sort -nr | head -4 2>&1 | tee >> $lfn
    echo ===== Memory in MB ========== 2>&1 | tee >> $lfn
    awk '{ print $NF }' $gpuStatsFile | sort -nr | head -4 2>&1 | tee >> $lfn

    echo ======== CPU Resources ========== 2>&1 | tee >> $lfn
    echo ===== Processor % ========== 2>&1 | tee >> $lfn
    awk '{ print $7 }' $cpuStatsFile | sort -nr | head -4 2>&1 | tee >> $lfn
    echo ===== Memory in KB ========== 2>&1 | tee >> $lfn
    awk '{ print $12 }' $cpuStatsFile | sort -nr | head -4 2>&1 | tee >> $lfn

fi

echo " ==6========== Evaluation for confidence threshold: $thrStr ================================================= " 2>&1 | tee >> $lfn

cd $cocoRoot

# setup detection results json file to pass it for evaluation
#detResultsJsonFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr""_iter_""$iterNum"".json"
detResultsJsonFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"
resultMatrixFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""resultMatrix"".log"
touch $resultMatrixFile
echo "---------- Start ----------" >> "$resultMatrixFile"
echo "Model : ""$modelFileName" >> "$resultMatrixFile"
echo "---------------------------" >> "$resultMatrixFile"

modelSrcDirFullPath="$modelsRootDir""/""$modelFileSrcDir"
echo "ModelFileSrcDir: $modelFileSrcDir" 2>&1 | tee >> $lfn
echo "modelSrcDirFullPath: $modelSrcDirFullPath" 2>&1 | tee >> $lfn
echo "detResultsJsonFile: $detResultsJsonFile" 2>&1 | tee >> $lfn

## 
#   Generate regular evaluation results only when prime_mAP is not required
##
if [ $generate_mAP_prime -eq 0 ]; then

    # setup directory to save images with detections annotated
    detectionImagesOutputDir="None"
    if [ "${createVideo}" -eq 1 ] ; then
        detectionImagesOutputDir="$workspaceDir""/""detections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
    fi
    resize_height=0
    resize_width=0

    echo "python PythonAPI/pycocoEvalDemo.py $gtFileName $dataSetDir $modelSrcDirFullPath $detResultsJsonFile $detectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg" 2>&1 | tee >> $lfn
    python PythonAPI/pycocoEvalDemo.py $gtFileName $dataSetDir $modelSrcDirFullPath $detResultsJsonFile $detectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg 2>&1 | tee -a >> $lfn $resultMatrixFile
    cd $caffeRoot/scripts/data_prep
    python result_matrix_formatter.py $resultMatrixFile 2>&1 | tee -a >> $lfn
    cd $cocoRoot
fi


# ============================================
#       generate detection and GT files with resized
#       bboxes and calc mAP again with these files
# ============================================
resizeBboxScript=$caffeRoot/scripts/data_prep/resizeBbox.py
pdetectionImagesOutputDir="None"

# Assume square 'input' for CNN, AND use the mAP_prime input value as resize height & width
if [ $generate_mAP_prime -ne 0 ]; then

    resize_height=$generate_mAP_prime
    resize_width=$generate_mAP_prime

    if [ "${createVideo}" -eq 1 ] ; then
        pdetectionImagesOutputDir="$workspaceDir""/""pdetections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
    fi

    #pdetResultsJsonFile="$workspaceDir""/""primed_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr""_iter_""$iterNum"".json"
    pdetResultsJsonFile="$workspaceDir""/""primed_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"

    echo " == 6A ========== Generating mAP Prime ============================================ " 2>&1 | tee >> $lfn

    echo "creating Primed GT Annotations & detections based on resized image into CNN" 2>&1 | tee >> $lfn
    echo "python $resizeBboxScript $gtFileName $pgtFileName $detResultsJsonFile $pdetResultsJsonFile $resize_height $resize_width" 2>&1 | tee >> $lfn
    python $resizeBboxScript $gtFileName $pgtFileName $detResultsJsonFile $pdetResultsJsonFile $resize_height $resize_width 2>&1 | tee >> $lfn

    echo "python PythonAPI/pycocoEvalDemo.py $pgtFileName $dataSetDir $modelSrcDirFullPath $pdetResultsJsonFilePrime $pdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg" 2>&1 | tee >> $lfn
    python PythonAPI/pycocoEvalDemo.py $pgtFileName $dataSetDir $modelSrcDirFullPath $pdetResultsJsonFile $pdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg 2>&1 | tee >> $lfn
fi

# ================================================================
#       calc mAP again with small, medium & large GT/DT files.
#               TODO: As time permitting :-)
# ================================================================
generate_split_results=0
if [ $generate_split_results -ne 0 ]; then

    echo " ==7A ============================================================ " 2>&1 | tee >> $lfn
    ##
    # TODO: Have to create these detection resutls file and only copy relevant detections
    #       Then, these can be used as is to evaluate for each size targets explicitly
    ##
    sdetResultsJsonFile="$workspaceDir""/""small_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"
    mdetResultsJsonFile="$workspaceDir""/""medium_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"
    ldetResultsJsonFile="$workspaceDir""/""large_""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"

    sgtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""small_""GT.json"
    mgtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""medium_""GT.json"
    lgtFileName="$dataRootDir""/""$dataSetDir""/""$dataTypeStr""/""large_""GT.json"

    resize_height=0
    resize_width=0

    sdetectionImagesOutputDir="None"
    mdetectionImagesOutputDir="None"
    ldetectionImagesOutputDir="None"
    if [ "${createVideo}" -eq 1 ] ; then
        sdetectionImagesOutputDir="$workspaceDir""/""small_detections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
        mdetectionImagesOutputDir="$workspaceDir""/""medium_detections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
        ldetectionImagesOutputDir="$workspaceDir""/""large_detections_""$dataSetDir2str""_catId_""$catId""_""$thrStr"
    fi

    ## using specific GT and Detection files
    ap_ar=-1
    area='all'
    echo "python PythonAPI/pycocoEvalDemo.py $sgtFileName $dataSetDir $modelSrcDirFullPath $sdetResultsJsonFile $sdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg" 2>&1 | tee >> $lfn
    python PythonAPI/pycocoEvalDemo.py $sgtFileName $dataSetDir $modelSrcDirFullPath $sdetResultsJsonFile $sdetectionImagesOutputDir $dataSetType $catId $viewImages $numberOfImagesToView $ap_ar $iouThr $area $confThr $resize_height $resize_width $maxIouThrArg $plotPRArg 2>&1 | tee >> $lfn

fi


# don't create tar file if we did not run the CNN model for evaluating detections
if [ "$skipRunDetEval" != 1 ]; then

    echo " ==7B ============================================================== " 2>&1 | tee >> $lfn
    echo "Bundling results under models directory into a single file." 2>&1 | tee >> $lfn
    myDir="$workspaceDir"
    cd $myDir

    # change to your needs : backup directory & result files being tar & zipped
    dateTimeStamp=$(date +"%Y%m%d%H%M%S")
    tarFileName="$dataSetDir2str""_""$dataTypeStr""_""catId""_""$catId""_""$thrStr""_""$dateTimeStamp"".tgz"
    filesToBeZipped="$dataSetDir2str""_""*.json ""$gpuStatsFile "" $dataSetDir2str""_""*.txt "" $lfn "" $cpuStatsFile ""$resultMatrixFile"
    echo "Run this command from $myDir to create a tgz file with all results" 2>&1 | tee >> $lfn
    echo "tar czvf $tarFileName $filesToBeZipped" 2>&1 | tee >> $lfn
    #tar czvf $tarFileName $filesToBeZipped 2>&1 | tee >> $lfn
    #echo "  Results File: $tarFileName "

    # change this to your needs
    #backupDir="/tmp"
    #tar --list -f $tarFileName 2>&1 | tee >> $lfn
    #cp $tarFileName $backupDir

    # change this to your needs: merge results*.json files into one json file for other uses
    # script to merge all json files only, to a single json file.
    # python mergeJsonFiles.py $results_directory $out_filename.json

fi

echo " ==8========================================================== "  2>&1 | tee >> $lfn

## call RoC / P-R scripts to plot if requested 
## Only do these if catId is > 0. 
## NOTE: coco style PR curves will be generated when catId is 0
if [ "${plotRoC}" -eq 1 ] || [ "${plotPR}" -eq 1 ] && [ "${catId}" -ne 0 ]; then

    myDir=$caffeRoot/scripts/data_prep
    cd $myDir

    echo "=========== PLOT RoC: $plotRoC P-R: $plotPR ====================" 2>&1 | tee >> $lfn
    echo "=========== PLOT RoC: $plotRoC P-R: $plotPR ====================" #2>&1 | tee >> $lfn

    #detResultsJsonFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr""_iter_""$iterNum"".json"
    detResultsJsonFile="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId""_""$thrStr"".json"
    detResultsJsonFilePrefix="$workspaceDir""/""$dataSetDir2str""_""$dataTypeStr""_""detectionResults_catId_""$catId"
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

echo " ==9========================================================== "  2>&1 | tee >> $lfn

if [ "${createVideo}" -eq 1 ] ; then
    if ! [ -x "$(command -v ffmpeg)" ]; then
        echo 'Error: ffmpeg is not installed.' 2>&1 | tee >> $lfn
        exit -1

    else

        if [ $generate_mAP_prime -eq 0 ]; then
            testVideoName="$workspaceDir""/""video_""$dataSetDir2str""_catId_""$catId""_""$thrStr"".mp4"
            ffmpeg -y -framerate 10 -pattern_type glob -i "$detectionImagesOutputDir""/""*.jpeg" $testVideoName 2>&1 | tee >> /dev/null
            echo "Test video available at $testVideoName" 2>&1 | tee >> $lfn

        else

            ## generate for prime case also
            ptestVideoName="$workspaceDir""/""pvideo_""$dataSetDir2str""_catId_""$catId""_""$thrStr"".mp4"
            ffmpeg -y -framerate 10 -pattern_type glob -i "$pdetectionImagesOutputDir""/""*.jpeg" $ptestVideoName 2>&1 | tee >> /dev/null
            echo "Test video for prime-mAP available at $ptestVideoName" 2>&1 | tee >> $lfn
        fi
    fi
fi

#echo " DONE!! Check Results File: $tarFileName & Logfile $lfn" 2>&1 | tee >> $lfn
echo " DONE!! Always good to look into the Logfile @  $lfn" 2>&1 | tee >> $lfn

echo "======================== END ======================" >> $lfn

echo "  Logfile $lfn"
echo " DONE!! "

