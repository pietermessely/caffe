#!/bin/bash

#-----------------------------------------------------------
# 2018-Q1: Augments for train / val LMDBs and generalization
#------------------------------------------------------------

#
# setup context from environment variables
# 
dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
modelsRootDir="${FLIR_MODELS_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"
if [ -z "$dataRootDir" ]; then
    echo "Need to setup ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi
if [ -z "$modelsRootDir" ]; then
    echo "Need to setup FLIR_MODELS_ROOT .... Continuing for now"
    #exit
fi
if [ -z "$caffeRoot" ]; then
    echo "Need to setup CAFFE_ROOT. Cannot Proceed Further"
    exit
fi
if [ -z "$cocoRoot" ]; then
    echo "Need to setup COCO_ROOT. Cannot Proceed Further"
    exit
fi

echo "data root: $dataRootDir"
echo "models root: $modelsRootDir"
echo "Caffe Root: $caffeRoot" 
echo "Coco Root: $cocoRoot" 

#
# get madatory input arguments - if none, bailout
#
dataSetDir=$1
if [ -z "$dataSetDir" ]; then
    echo "Need to provide data Set Directory relative to ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi
echo "data set dir is $dataSetDir"

labelMapFile=$2
if [ -z "$labelMapFile" ]; then
    echo "Need to provide Label Map File. Cannot Proceed Further"
    exit
fi
echo "Label Map File is $labelMapFile"

####################################################################################3333
# Other input args needed : will get to these later as necesasry
#if [ "$#" -lt "1" ]
#   then
#	echo "Need Arguemnts [in specific order]:
#	root-dir-of-dataset*: e.g., /mnt/data/datasets/filrdata2
#	anno-type: e.g., detection, which is the default
#	map_file: e.g., labelMapFile/labelmap_coco.prototxt, default is data/coco/labelmap_coco.prototxt
#	database type: e.g., lmdb, which is the default
#	output_dir_for_Databases: e.g., lmdb, which is the default
#	example_dir_to_store_database_links: e.g., examples, which is the default " 
#	echo " "
#	echo "Note: * Indicates Mandatory Argument"
#   exit 1
#fi

# check and gather the other inputs
# annotation type
#if [ "$#" -ge "3" ]
#   then
#	anno_type="$3"
#fi
#
## database type 
#if [ "$#" -ge "4" ]
#   then
#	db="$4"
#fi
#
## output dir for databases 
#if [ "$#" -ge "5" ]
#   then
#	train_outdir=$(printf "%s/%s" "$train_root_dir" "$5")
#	test_outdir=$(printf "%s/%s" "$test_root_dir" "$5")
#fi
#
## output dir for examples
#if [ "$#" -ge "6" ]
#   then
#	train_example_dir=$(printf "%s/%s" "$train_root_dir" "$6")
#	test_example_dir=$(printf "%s/%s" "$test_root_dir" "$6")
#fi
#
####################################################################################3333

label_type="xml"
prefix="$dataRootDir""/""$dataSetDir"

train_root_dir=$(printf "%s/train" "$prefix")
val_root_dir=$(printf "%s/val" "$prefix")

# setup the defaults - will be under the root dir
db="lmdb"
anno_type="detection"
min_dim=0
max_dim=0
width=0
height=0
redo=1

# setup other required variables : hardcoded for now
#extra_cmd="--encode-type=jpeg --encoded"
extra_cmd="--encode-type=jpeg --encoded --shuffle"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

# setup other contexts
annoScript="$caffeRoot""/scripts/create_annoset.py"
#dir2linkLmdbs="$caffeRoot""/examples"
dir2linkLmdbs="examples"

#prefix="$dataRootDir""/""$dataSetDir"
#valListFile="$prefix""/valList.txt"
#valLmdbDir="$prefix""/""$dataSetDir"".lmdb"
#valLogFile="$prefix""/""$dataSetDir""_create_data_script.log"

train_example_dir=$(printf "%s/examples" "$train_root_dir")
train_outdir=$(printf "%s/%s" "$train_root_dir" "$db")
train_list_file=$(printf "%s/trainList.txt" "$train_root_dir")

val_example_dir=$(printf "%s/examples" "$val_root_dir")
val_outdir=$(printf "%s/%s" "$val_root_dir" "$db")
val_list_file=$(printf "%s/valList.txt" "$val_root_dir")

#
# The arguments in order are: 
# root :  directory which contains the dataset images and annotations
# list file : this has the image & annotation paths (output of create_list_split.py)
# outdir : where the output database file is written
# example dir: stores the link of the database files
# rest with "--option-foo" is self explainatory
# 

# first call is for setting up the train dataset
echo "Setting up train dataset "

echo "# python $annoScript --anno-type=$anno_type --label-type=$label_type --label-map-file=$labelMapFile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $train_list_file $train_outdir $train_example_dir " 

python $annoScript --anno-type=$anno_type --label-type=$label_type --label-map-file=$labelMapFile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $train_list_file $train_outdir $train_example_dir

# Now setup the validation test dataset
echo " Setting up validation test dataset "
echo "# python $annoScript --anno-type=$anno_type --label-type=$label_type --label-map-file=$labelMapFile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $val_list_file $val_outdir $val_example_dir "

python $annoScript --anno-type=$anno_type --label-type=$label_type --label-map-file=$labelMapFile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $val_list_file $val_outdir $val_example_dir 


