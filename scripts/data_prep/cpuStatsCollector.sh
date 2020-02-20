#!/bin/bash
#
# to get CPU & Memory resource usage stats
# input arguments (1) output filename (2) number of snapshots to take (1..999)
#


# Interval in Seconds for each snapshot (1..xx seconds) & count are the number of snapshots
interval=1 
outfilename=$1
count=$2

echo "pidstat invoked with interval: $interval, count: $count, output_file: $outfilename"
#pidstat -G FLIR -h -u -r -l -s $interval $count > $1
pidstat -G FLIR -h -u -r -s $interval $count > $1



