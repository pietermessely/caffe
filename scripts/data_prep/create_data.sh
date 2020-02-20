cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

#
# setup context from environment variables
# 
#testDir="${FOO_DIR:-""}"
#echo $testDir
dataRootDir="${ML_TEST_DATA_SET_ROOT:-""}"
caffeRoot="${CAFFE_ROOT:-""}"
cocoRoot="${COCO_ROOT:-""}"
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

echo "data root: $dataRootDir"
echo "Caffe Root: $caffeRoot" 
echo "Coco Root: $cocoRoot" 

dataSetDir=$1
if [ -z "$dataSetDir" ]; then
    echo "Need to provide data Set Directory relative to ML_TEST_DATA_SET_ROOT. Cannot Proceed Further"
    exit
fi
echo "data set dir is $dataSetDir"

# FIXME: 
#       Only Validation dataset supported. Extend for Train/Test also.
#       Below hard-coded variables for now
#
#data_root_dir=$cur_dir
data_root_dir=$dataRootDir
dataset_name="coco"
label_type="xml"
#mapfile="/mnt/data/ssd_clean/caffe/data/$dataset_name/labelmap_voc.prototxt"
#mapfile="/mnt/data/ssd_clean/caffe/data/$dataset_name/labelmap_coco.prototxt"
#mapfile="/home/flir/warea/ub-setup/caffe/data/coco/labelmap_coco.prototxt"
#mapfile="/home/flir/warea/ub-setup/caffe/scripts/data_prep/labelmap_flir.prototxt"
#mapfile="/home/flir/warea/ub-setup/caffe/scripts/data_prep/labelmap_coco.prototxt"

# TODO: labelmap_coco.prototxt required. This should be the same as used for training
mapfile="$caffeRoot/scripts/data_prep/labelmap_coco.prototxt"
echo "mapfile: $mapfile"

anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
redo=1

# doing only for val. Augment to support test/train DSs also.
dataSetType=1   
if [ "$dataSetType" == 1 ]; then
    dataTypeStr="test"
fi

if [ "$dataSetType" != 1 ]; then
    echo "dataSet Type $dataSetType NOT supported yet."
    exit
fi

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

# setup other contexts
annoScript="$caffeRoot""/scripts/create_annoset.py"
prefix="$dataRootDir""/""$dataSetDir""/""$dataTypeStr"
testListFile="$prefix""/testList.txt"
testLmdbDir="$prefix""/""$dataSetDir"".lmdb"
testLogFile="$prefix""/""$dataSetDir""_create_data_script.log"
dir2linkLmdbs='examples'

#
# NOTE: list.txt has the full path for image and annotation files. 
#       So, data_root_dir is not requried
#

echo "python $annoScript --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $testListFile $testLmdbDir $dir2linkLmdbs 2>&1 | tee $testLogFile "

  python $annoScript --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / $testListFile $testLmdbDir $dir2linkLmdbs 2>&1 | tee $testLogFile

##  python /mnt/data/ssd_clean/caffe/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd /mnt/data/dronedata list.txt /mnt/data/dronedata/dronedata.lmdb examples 

##  python /home/flir/warea/ub-setup/caffe/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd /home/flir/warea/ub-setup/caffe/data/dronedata /home/flir/warea/ub-setup/caffe/data/dronedata/list.txt /home/flir/warea/ub-setup/caffe/data/dronedata/dronedata.lmdb examples 2>&1 | tee $root_dir/data/dronedata/dronedata.log

#  python /home/flir/warea/ub-setup/caffe/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --label-type=$label_type --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd / /home/flir/warea/ub-setup/caffe/data/aporia-cars-nov9/val/valList.txt /home/flir/warea/ub-setup/caffe/data/aporia-cars-nov9/val/aporia-cars-nov9.lmdb examples 2>&1 | tee $root_dir/data/aporia-cars-nov9/val/val-aporia-cars-nov9.log
