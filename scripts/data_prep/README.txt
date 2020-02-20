This README will outline the steps required to run detection evaluation tests.

If you are reading this file, you must have "$CAFFE_ROOT/scripts/data_prep" directory.
In this directory are a set of scripts to help prepare your datasets for evaluating your 
    trained models and post the results in the same directory as the model file. The scripts
    in coco/PythonAPI are also used in generating evaluation metrics.


PREPARE
--------

(0) coco_labels.json & deploy.prototxt: Scripts expect these files to be in the same 
directory as the models file. 

(1) Clone caffe and coco from github and build them. The test scripts use ssd_detect_FLIR
    binary in $CAFFE_ROOT/build/examples/ssd directory.
e.g.,   git clone git@github.com:FLIR/coco.git -b dev
        git clone git@github.com:FLIR/caffe.git -b ssd
    Build caffe and coco. There is a 'build.sh' script in each directory.
  
(2) Download "mytestdata" you need for testing, typically from conservator

(3) Populate "models" from git-lfs to get the model files for testing. Requires git-lfs (https://git-lfs.github.com).
    Alternatively, you can use this script to test a model you trained and available locally on your system

(4) For the scripts to run, the following environment variables must be set and 
ensure all necesasry files exist (model files, dataset directory, etc.,).

    CAFFE_ROOT (root directory of caffe code e.g., /home/flir/warea/wperf/caffe) 
    COCO_ROOT (root directory of coco code e.g., /home/flir/warea/wperf/coco) 
    ML_TEST_DATA_SET_ROOT (data set root directory e.g., /mnt/data/testdataset) 
    FLIR_MODELS_ROOT (models root directory e.g., /mnt/data/models) 
    PYTHONPATH (should include paths to: $CAFFE_ROOT/python:$COCO_ROOT/PythonAPI:$CAFFE_ROOT/scripts/data_prep)


RUN TESTS
----------

Run the testScript.sh bash script in caffe/scripts/data_prep directory
    - With no input arguments to the script, it will do evaluation based on defaults.
        The results files will be in the same dir as the model file being tested".
        The following variables have defaults within this script. NOTE: The default values
        in the script that point to datasets, model files, label files etc., may not be applicable on
        your system. 

    echo "using the following to test the model"
    echo "dataSetDir                                   : $ML_TEST_DATA_SET_ROOT/$dataSetDir "
    echo "dataSetType(1: test, 0: train_val)           : $dataSetType"
    echo "modelFileSrcDir                              : $FLIR_MODELS_ROOT/$modelFileSrcDir "
    echo "modelFileName                                : $modelFileSrcDir/$modelFileName "
    echo "cnnUnderTest                                 : $cnnUnderTest "
    echo "                                             : (e.g., caffe/examples/ssd/ssd_detect_FLIR)
    echo "categoryId_to_check_results (o: all)         : $catId"
    echo "skipCreateGT (1:yes, 0: No)                  : $skipCreateGT"
    echo "skipRunDetEval (1:yes, 0: No)                : $skipRunDetEval"
    echo "viewImages(1:yes, 0: No)                     : $viewImages"
    echo "numberOfImagesToView                         : $numberOfImagesToView"
    echo "confidence_threshold (0.01 ... 0.95)         : $conf_thr"
    echo "batch_size(1..512)                           : $batch_size"
    echo "ap_ar (1:AP, 0: AR, -1: ALL)                 : $ap_ar"
    echo "area (small, medium, large, all)             : $area"
    echo "iouThr (0: ALL, 0.5, 0.75)                   : $iouThr"
    echo "labelMapFile                                 : $labelMapFile"
    echo "logFile (tail -f logFile for progress)       : $lfn"
    echo "plotRoC curve (1: yes, 0: no)                : $plotRoC"
    echo "plotPR  curve (1: yes, 0: no)                : $plotPR"
    echo " "
    echo "NOTE: {CAFFE,COCO,ML_TEST_DATA_SET,FLIR_MODELS}_ROOT & PYTHONPATH must be set properly"

    - script can be invoked to selectively do evaluations or look at prior results, etc.,
        These are variables within the testScript.sh currently 

Input Arguments: 
----------------

# ===================================================================================
#
# All arguments are optional. 
#       If present, SHOULD be in the FORMAT as noted below.
#       When NOT present, default values will be used
#
# Format of Input Args are: 
#
#       -a: mAP Prime:
#               [0..1] : 0 (default) disables calculating mAP-Prime. 1 enables this.
#                        These are stored in the corresponding json files and also
#                        dumped into the log file.
#       -b: batch_size: 
#               [1..512] : depending on your system, higher values may result in errors
#
#       -c : Category-ID: 
#               [0..90], where '0' is ALL and 'x' is mapped coco ID.
#               This will be used to gather/calcuate the test results.
#
#       -d : test_data_set_directory: 
#               relative path from ML_TEST_DATA_SET_ROOT
#               e.g., input "mytestdataset" dir should be at "$ML_TEST_DATA_SET_ROOT/mytestdataset"      
#
#       -e: Skip_Detection_Eval: 
#               [0/1] : 0 is don't skip and 1 is skip
#
#       -f : Model_File_Name: 
#               model file to be used. Expected to be in "$Model_File_Dir"
#
#       -i: view_detection_results_on_images:
#               [0..9999] : 0 is no and +ve number is used as number of images to show.
#                Spawns images with GT & DT bboxes
#
#       -m : Model_File_Dir: 
#               relative path from FLIR_MODELS_ROOT
#               e.g., input "object_detectors/aerial/ssd_vgg_512x512" dir 
#                       should be at  
#               "$FLIR_MODELS_ROOT/object_detectors/aerial/ssd_vgg_512x512"      
#
#       -p: plot ROC & PR Curves:
#               [0/1/2/3], where  0: No Plot, 1: ROC Plot, 2: PR Plot, 3: both
#
#       -s: Skip_GT: 
#               [0/1] : 0 is don't skip and 1 is skip
#
#       -z: create detection output Video: 
#               [0/1] : 0 will not create a video and 1 is creates the video from detection output images.
#                       These images have both the GT and Detection bounding boxes with Id and
#                       scores from the model under test. 
#               TODO: Output Images are saved in test dataset directory under 
#                       'Detections' & 'pdetections' folder for the corresponding
#                       calcs by default. We need to do this as part of video creation 
#                       option too and clean up existing directories before saving.
#
#       e.g., ./testScript.sh -m "object_detectors/aerial/ssd_vgg_512x512" \
#                                -f "aerial_ssd_vgg_512x512_iter_11000.caffemodel" \
#                                -d "mytestdataset"
# ===================================================================================


 
Other Notes:
------------
- createList.py : enable code to map 'cow:20' to a drone object type, if you are using the
                    "dronedata" dataset from conservator
- testScript.sh: 
        - if {CAFFE,COCO,ML_TEST_DATA_SET,FLIR_MODELS}_ROOT any are not set, script will bailout 
        - Check for text "change to your needs" to modify per your needs 
        - dataSetDir: The default in the script is an invalid one & will fail.
        - catId: default is '0'. So, AP & AR scores may be skewed 
        - skipCreateGT, skipRunDetEval : you can set these to skip those steps
        - viewImages: enable this to spawn a viewer with the detection results
        - gpuStats: gpu usage snapshot using an nvidia tool, gives gpu & memory stats
        - cpuStats: cpu usage snapshot using pidtstat tool
        - batchSize: default is 1. Based on your GPU memory, this can be changed.
                     On an NVIDIA 1080 with 8GB of memory, VGG_SSD_512x512, a batch 
                     size of '13' is the max before you see 'memory errors'. 
        - lfn: the default log file is /tmp/$USER/k.log and is always appended.

- data augmentation: The data augmentation that caffe does (data layer) to the input images can
        be viewed. By default, 50 images will be saved in "/mnt/data/caffe_data_augments" directory
        if this directory exists. Every starting session would overwrite this set. To enable
        a snapshot at an interval, look for 'debug hack' and change the 'if' conditional in the
        CAFFE_ROOT/src/caffe/data_transform.cpp file.

- mAP for specific class: The variable to enable this can be included in your training script to
        get a dump of per class mAP during the validation_tests done as part of the training. To
        enable this, include the following in solver_param: 'show_per_class_result': True,
        NOTE: src/caffe/solver.cpp has been modified to also print the mAP based only on the 
            number of labels with TP outputs rather than using the number_of_classes/labels. 

The relevant results after each run are zipped into a '.tgz' file, with a datetime stamp
appended to filename for ease of use.
 
Tips:
------
Typically, scripts checked into github will work without issues if the above guidelines are 
followed and met. Don't go chasing into the scripts on failures. Best is to look into the log
file that is generated to catch any errors. Understanding the terminal output would help 
with failures. 

