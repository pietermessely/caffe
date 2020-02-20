'''
Q2 2019
@ethomson
Goal is to be able to convert annotations from any format we use to any other
format we use.
========================
Supported input formats:
    * detection log
        - Output of running a cnn. One big txt/log file with one detection per line
    * coco
        - Coco with a unique json per image
    * xml
        - Format of a dataset downloaded from conservator. Also the VOC format.
    * conservator
        - Format we use to upload annotations for a video to conservator. One big json file
          https://www.flirconservator.com/help
    * unified coco
        - coco with all annotations in one big json file
          http://cocodataset.org/#format-data
Supported output formats:
    * coco
    * detection log
    * conservator
    * xml
========================

Example run command:
$ python anno_converter.py xml coco --src_dir ../test_280HD-drone_video-snippets/ --save_location ../recurseive_read_coco --catids /mnt/fruitbasket/label_files/flir_catids.json --labelvector /mnt/fruitbasket/label_files/flir_labelvector.json

TODO:
Input: (lmdb/VEDAI/uav?)
Output: (lmdb?) unified_coco
- view while converting
- Convert image types? (Ex: all to jpeg)
'''

from __future__ import print_function

from dataset import Dataset
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("from_format", type=str, help="Format to read annotations from", choices=['detlog', 'xml', 'coco', 'conservator', 'unified_coco'])
    parser.add_argument("to_format", type=str, help="Format to convert annotations to", choices=['detlog', 'xml', 'coco', 'conservator, unified_coco'])
    parser.add_argument("-d", "--src_dir", type=str, help="Location to load data from (reading coco/xml/conservator data)")
    parser.add_argument("-i", "--input_file", type=str, help="Location of input file. Needed for loading from detlog and conservator.")
    parser.add_argument("-s", "--save_location", type=str, help="Location to save resulting files", required=True)
    parser.add_argument("-p", "--prefix_dir", type=str, help="Name of folder to find images in (Data/PreviewData)", default="Data")
    parser.add_argument("--remap_catids", type=str, help="json file specifying remapping of cat ids")
    parser.add_argument("--remap_labels", type=str, help="json file specifying remapping of annotation labels")
    parser.add_argument("--filter_catids", type=int, nargs='*', help="List of catids to filter out. Filters after remapping.")
    parser.add_argument("-c", "--catids", type=str, help="Full path to catid json file")
    parser.add_argument("-l", "--labelvector", type=str, help="Full path to labelvector json file")
    parser.add_argument("-e", "--email", type=str, help="Your email. Needed only for saving in conservator format")
    parser.add_argument("-f", "--flatten_filestruct", action="store_true", help="Don't preserve file structure of images. Save all images to one folder.")
    parser.add_argument("-n", "--dataset_name", type=str, help="Name of dataset")
    args = parser.parse_args()

    from_format = args.from_format
    output_format = args.to_format
    input_file = args.input_file
    save_root = args.save_location
    src_dir = args.src_dir
    remap_catids = args.remap_catids
    remap_labels = args.remap_labels
    catids_map_fname = args.catids
    labelvector_fname = args.labelvector
    dataset_name = args.dataset_name
    prefix_dir = args.prefix_dir
    email = args.email
    catid_filter = args.filter_catids
    flatten_filestruct = args.flatten_filestruct

    dataset = Dataset(name=dataset_name, catids_map=catids_map_fname, labelvector=labelvector_fname, email=email)

    print("Loading...")
    if from_format == 'detlog':
        if not input_file:
            raise Exception("ERROR: To load from a detlog file, use '--input_file' to specify the log file.")
        dataset.load_detlog(input_file)
    elif from_format == 'coco':
        if not src_dir:
            raise Exception("ERROR: To load from coco, use '--src_dir' to specify the directory containing data & annotations")
        dataset.load_coco(src_dir, prefix_dir)
    elif from_format == "xml":
        if not src_dir:
            raise Exception("ERROR: To load from xml, use '--src_dir' to specify the directory containing data & annotations")
        dataset.load_xml(src_dir, prefix_dir)
    elif from_format == "conservator":
        if not src_dir or not input_file:
            raise Exception("ERROR: To load from conservator, use both options '--src_dir' and '--input_file'")
        dataset.load_conservator(src_dir, input_file)
    elif from_format == "unified_coco":
        if not src_dir or not input_file:
            raise Exception("ERROR: To load from unified_coco, use both options '--src_dir' and '--input_file'")
        dataset.load_unified_coco(src_dir, input_file)
    else:
        raise Exception("ERROR: Invalid load format ({})".format(from_format))

    print(dataset)

    print("Remapping...")
    if remap_labels is not None:
        dataset.remap_labels(remap_labels)
    if remap_catids is not None:
        dataset.remap_catids(remap_catids)

    print("Sanitizing...")
    print("Filtering out category ids: {}".format(catid_filter))
    dataset.sanitize(catid_filter=catid_filter)

    print("Saving...")
    dataset.save(save_root, output_format, flatten=flatten_filestruct)
