#!/usr/env python

import json
import sys
import os

# ============================================================================================
# Note: 
#       This can be invoked by-itself or is invoked by evaluationScript.sh when commanded.
#
# Input Arguments:
# ----------------
#
# annoSet.json:  This is annotation_set.json file in your test dataset directory (conservator generated)
# detection_log: This is the detections output from your model, on images from image_Set.json
#                Each detection is a specific set of values formatted as follows:
#                <image_file> <category_id> <score> <xmin> <ymin> <xmax> <ymax> <time1> <time2>
#                where, time1 & 2 are net->forward and eval times. Set to 1 if not available
# [prefix]     : This is the path prefix that would be prepended to the image_file in each
#                detection, pointing to the actual/absolute location in the test dataset.
#                       e.g., : "/mnt/data/flir-data/mytestdataset/test/Data"
#
# Output:
# -------
#       Top Secret encodings with proper paths to evaluate against the GT, in a spearate file
#
# ============================================================================================

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print "usage: %s <annoSet.json> <detection_log> [prefix]" % sys.argv[0]
    sys.exit(1)

filenames = json.load(open(sys.argv[1], 'r'))
filename_ids = {}
num2print=False
toPrint=0

# setup ids for filenames in annoSet.json 
for i, filename in enumerate(filenames):

    # do this with just the file names. We will pre-pend the input path/prefix later.
    fpath, ffname = os.path.split(filename)
    fname, fext = os.path.splitext(ffname)
    # remove extension also - we are looking into annotation set file
    filename_ids[fname] = i
    if num2print is True:
        print 'filename: %s fname: %s, i: %d'%(filename, fname, i)
        print filename_ids[fname]
        num2print = False


# setup output file based on input file : just change the extension. 
inFile, inFileExt = os.path.splitext(sys.argv[2])
outputFile = open('%s.log'%(inFile), 'w')

not_in_dataset = []
for line in open(sys.argv[2]).readlines():
    # line: <filename-maybe-with-spaces> <w1> <w2> ... <w8>

    # join first N-8 words to handle case where filename has spaces, and concatenate with the rest
    words = line.split(" ")
    words = [" ".join(words[:-8])] + words[-8:]
    assert len(words) == 9

    orig_path = words[0]
    opath, oofname = os.path.split(orig_path)
    ofname, oext = os.path.splitext(oofname)

    # check if filename exists - must be unique by itself
    #if not orig_path in filename_ids:
    if not ofname in filename_ids:
        not_in_dataset.append(orig_path)
        if num2print is False:
            num2print = True
            print orig_path
            print ofname + ' ' + oext
        continue

    # setup ID for each filename and also add prefix if provided
    filename_id = filename_ids[ofname]
    if len(sys.argv) == 4:
        prefixed_filename = "%s/%s%s" % (sys.argv[3], ofname, oext)
        if toPrint < 0 :    # for debug 
            print 'prefixed filename: ' + prefixed_filename
            toPrint += 1
    porig_name, pext = os.path.splitext(prefixed_filename)
    new_name = "%s-%d%s" % (porig_name, filename_id, pext)
    words[0] = new_name
    #sys.stdout.write(" ".join(words))
    outputFile.write(" ".join(words))

outputFile.close()

if len(not_in_dataset) > 0:
    sys.stderr.write("Note: Found %d files in log that were not in the dataset\n" % len(not_in_dataset))
    #sys.stderr.write("\n".join(not_in_dataset))


