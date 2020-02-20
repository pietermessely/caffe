#!/bin/bash
#
# to get nvidia based GPU stats
# input arguments (1) output_file.csv, (2) snapshot timer in milliSeconds
#
# We can also initiate this in daemon mode. See commands way below. Not used
#       since it does not have some of the stats we need.
#


#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_utillization.log; 
#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader -f "test.csv" >> gpu_utillization.log; 
#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pcie.link.width.current,pstate,encoder.stats.averageFps,encoder.stats.averageLatency,ecc.errors.uncorrected.aggregate.total --format=csv -f "test.csv" 
#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pcie.link.width.current,pstate,encoder.stats.averageFps,encoder.stats.averageLatency,ecc.errors.uncorrected.aggregate.total --format=csv -f "test.csv" -lms 100

# this one loops forever or ctrl-c, with snapshot output every 100ms

#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pcie.link.width.current,pstate,encoder.stats.averageFps,encoder.stats.averageLatency,ecc.errors.uncorrected.aggregate.total --format=csv,nounits -f "test.csv" -lms 100

#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,pcie.link.width.current,pstate,encoder.stats.averageFps,encoder.stats.averageLatency --format=csv,nounits -f "test.csv" -lms 100

# input arguments (1) output_file.csv, (2) snapshot timer in milliSeconds
echo "Output Reuslts File: $1 & Loop Timer: $2 msecs"

nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used, --format=csv,nounits -f $1 -lms $2


#
# This to run a daemon, but the snapshot is only in seconds.
#
# gather memory, utili, throughput and error stats, tag the output stats file with 'test1'
#sudo nvidia-smi daemon -d 1 -s mute -j "test1"

# terminate the daemon (do this eventually. Don't need to, just to get the stats)
#sudo nvidia-smi daemon -t

# replay the stats onto the stdout ... takes for eeeeevvveeerrr .i.e 1s/Line snapshot interval
#sudo nvidia-smi replay -s mute -f /var/log/nvstats/nvstats-20171219-test1 

# replay the stats into a readable file : finishes in a sec or two.
#sudo nvidia-smi replay -s mute -f /var/log/nvstats/nvstats-20171219-test1 -r /tmp/k1.txt
#

