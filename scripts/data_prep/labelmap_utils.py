#!/usr/env/python
from __future__ import print_function

import click
import os
import json
import sys

# This function retrieves the COCO category ids from the name. Here COCO refers to a *format*, not a
# particular set of labels.
#
# The format of coco_catids_filename is generated as follows.
# Starting with this:
#   https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
# We get COCO categories like this:
#   cats = coco.loadCats(coco.getCatIds())
# And save them to JSON like this
#   json.dumps(open(outfile, 'w'), cats)
# This function reads that JSON file and returns a mapping of label name to COCO id
def get_name_to_coco_catid(coco_catids_filename):
    # Load annoids in COCO format
    catids_json = json.load(open(coco_catids_filename, 'r'))
    name_to_catid = {}
    for catid in catids_json:
        name_to_catid[catid["name"]] = catid["id"]
    return name_to_catid

# Loads a FLIR labelvector file
def get_labelvector(labelvector_filename):
    labelvector_json = json.load(open(labelvector_filename, 'r'))
    if not "values" in labelvector_json:
        raise Exception("Did not find 'values' field in labelvector json.")
    return labelvector_json["values"]

# Loads name-to-index mapping for a FLIR labelvector file
def get_name_to_labelvector_id(labelvector_filename):
    labelvector = get_labelvector(labelvector_filename)
    return dict([(name, i) for i, name in enumerate(labelvector)])

# Creates a labelmap prototxt that describes the mapping from annotation IDs (specified in
# COCO format) to labelvector indices.
def create_labelmap_prototext(coco_catids_filename, labelvector_filename, labelmap_prototxt_filename):
    name_to_coco_catid = get_name_to_coco_catid(coco_catids_filename)
    labelvector = get_labelvector(labelvector_filename)

    lines = []
    for labelid, label in enumerate(labelvector):
        if labelid == 0:
            if not label in set(["empty", "background", "none_of_the_above", "nothing"]):
                print("WARNING: Element 0 is being mapped to background but does not appear to have a background-like name: %s" % label)
            #name = "none_of_the_above"
            name = "0" # jk
            display_name = "background"
        else:
            if not label in name_to_coco_catid:
                print("WARNING: Target label not found in source coco_catids: %s" % label)
                continue
            coco_catid = name_to_coco_catid[label]
            name = str(coco_catid)
            display_name = label

        lines.append("item {")
        lines.append("  name: \"%s\"" % str(name))
        lines.append("  label: %d" % labelid)
        lines.append("  display_name: \"%s\"" % display_name)
        lines.append("}")

    out_file = open(labelmap_prototxt_filename, 'w')
    out_file.writelines("\n".join(lines) + "\n")

@click.command()
@click.argument("coco_catids_json")
@click.argument("labelvector_json")
@click.argument("labelmap_prototxt")
def cli(coco_catids_json, labelvector_json, labelmap_prototxt):
    create_labelmap_prototext(coco_catids_json, labelvector_json, labelmap_prototxt)

if __name__ == "__main__":
    cli()
