"""

Q4 2017:        Created to merge multiple json files into one
@author:        kedar m

"""

'''
    This is to create one output results file
'''

import os
import json
import shutil
import string
import sys
import argparse

# input arguments 
parser = argparse.ArgumentParser()
parser.add_argument("resultsDir", type=str,
                    help="Directory where the json files exist. Absolute Path")

parser.add_argument("outFname", type=str,
                    help="Output json filename, to be created in the directory passed above")

# get the inputs and prep the output fname
args = parser.parse_args()
indir = args.resultsDir
ofn = args.outFname
ofname = '%s/%s'%(indir, ofn)

# check if the dir exists and or valid
print 'results Diretory: {}'.format(indir)
if not os.path.isdir(indir):
    print 'Provided input argument resultsDir does NOT exist or in not a directory! {}'.format(indir)
    exit

# get the list of files. Skip the coco_labels.json if it exists and previous results files 
flist = []
for fname in os.listdir(indir) :
    name, ext = os.path.splitext(fname)
    #print 'file {} name {} ext {}'.format(fname, name, ext)

    # skip if this is the results file ir coco_label file
    if fname == ofn : 
        continue
    if name == 'coco_labels' :
        continue

    if ext == '.json' :    
        lfname = '%s/%s'%(indir, fname)
        flist.append(lfname)
    
#print 'input File List to merge: {}'.format(flist)


def mangle(s):
    print s.strip()[1:-1]
    return s.strip()[1:-1]

# read each file and write into the destination file
with file(ofname, "w") as ofile:

    first = True
    pfirst = False

    for ifname in flist:

        with file(ifname) as ifile:

            if first:
                ofile.write('{ \n')
                first = False
            else:
                ofile.write(', ')
                #ofile.write('\n\t')

            s = ifile.read()
            s1 = s.strip()[1:-1]
            if pfirst:
                print s[0]
                print s[-1]
                pfirst = False


            # depending on json being a single list or multiple lists, write output
            if s[0] == '[' :
                # get the file name without extension and use it as list name
                ipath,iname = os.path.split(ifname)
                hdr,ext = os.path.splitext(iname)
                odata = '  "' + str(hdr) + '" : ' + s 
            else :
                odata = s1
            
            # dump odata 
            #ofile.write(mangle(ifile.read()))
            ofile.write(odata)

    ofile.write('}')
    ofile.write('\n')

print 'Merged all the json files in the directory {} into the file {} '.format(indir, ofname)

