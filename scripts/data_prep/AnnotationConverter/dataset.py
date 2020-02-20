from __future__ import print_function

from collections import defaultdict, OrderedDict
from annotation import Annotation
from image import Image
from PIL import Image as pimage
import xml.etree.ElementTree as ET
import os.path as osp
import labelmap_utils
import fnmatch
import json
import re
import os


ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.png', '.tiff', '.jpeg', '.gif', '.bmp']
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]


class Dataset(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.images = []
        self.metadata = kwargs.get('metadata', "")
        self.version = None
        self.date = None
        self.catids_map = None
        catids_fname = kwargs.get("catids_map", None)
        if catids_fname:
            self.catids_map = labelmap_utils.get_name_to_coco_catid(catids_fname)
        self.labelvector = None
        lv_fname = kwargs.get('labelvector', None)
        if lv_fname:
            self.labelvector = labelmap_utils.get_name_to_labelvector_id(lv_fname)
        self.catid_to_lvid = None
        if self.catids_map and self.labelvector:
            self.catid_to_lvid = {}
            for name, catid in self.catids_map.items():
                if name not in self.labelvector.keys():
                    print("WARNING: name ({}) is in catids map but not labelvector".format(name))
                self.catid_to_lvid[catid] = self.labelvector.get(name, -1)
        self.catid_to_name = None
        if self.catids_map:
            self.catid_to_name = dict(reversed(item) for item in self.catids_map.items())
        self.email = kwargs.get('email', None)

    def sanitize(self, catid_filter=None):
        for image in self.images:
            image.sanitize(catids_map=self.catids_map, catid_filter=catid_filter, catid_to_name=self.catid_to_name)

    def remap_catids(self, mapping_filename):
        catid_map = json.load(open(mapping_filename, 'r'))
        for image in self.images:
            image.remap_catids(catid_map)

    def remap_labels(self, mapping_filename):
        label_map = json.load(open(mapping_filename, 'r'))
        for image in self.images:
            image.remap_labels(label_map)

    def save(self, output_root, format, flatten):
        """
        Save the dataset to output_root as type format
        """
        if format == 'coco':
            """
            Save dataset in coco format.
            Produces output_root/Data which contains the images
            and      output_root/Annotations which contains an annotation json file for each image.
            """
            image_save_dir = osp.join(output_root, 'Data')
            anno_save_dir = osp.join(output_root, 'Annotations')
            # Not thread safe
            if not osp.exists(image_save_dir):
                os.makedirs(image_save_dir)
            if not osp.exists(anno_save_dir):
                os.makedirs(anno_save_dir)

            for image in self.images:
                # save image file
                image.save_image(image_save_dir, flatten)
                # save annotation
                image.save_coco(anno_save_dir, flatten)

        elif format == "conservator":
            """
            Save dataset in conservator metadata json format.
            Produces output_root/{dataset_name}.json
            """
            if not self.name:
                raise Exception("Need to give a name to save in conservator format. (Use --dataset_name)")
            if not self.email:
                raise Exception("Need to give an email to save in conservator format. (Use --email)")

            metadata = OrderedDict([
                        ('version', 1),
                        ('overwrite', True),
                        ('videos', [OrderedDict([
                                        ('name', self.name),
                                        ('description', self.metadata),
                                        ('owner', self.email),
                                        ('location', 'Unknown'),
                                        ('frames', [])
                                    ])])
                        ])

            frames = []
            # Sorts using natural_sort_key on image name
            for frame_num, image in enumerate(sorted(self.images)):
                frame_annos = OrderedDict([
                                ('frameIndex'  , frame_num),
                                ('annotations' , [])
                              ])
                for anno in image.annotations:
                    boundingBox = OrderedDict([
                                    ('x', anno.l),
                                    ('y', anno.t),
                                    ('w', anno.w),
                                    ('h', anno.h)
                                ])
                    disp_name = anno.name
                    if anno.name is None:
                        disp_name = self.catid_to_name.get(anno.cat_id, None)
                    if disp_name is None:
                        raise Exception("Anno has no name and anno's catid not in catids")
                    converted_anno = OrderedDict([('labels'      , [disp_name]),
                                                  ('boundingBox' , boundingBox)
                                     ])
                    frame_annos['annotations'].append(converted_anno)
                frames.append(frame_annos)
            metadata['videos'][0]['frames'] = frames

            if not osp.exists(output_root):
                os.makedirs(output_root)
            outfile_fp = os.path.join(output_root, self.name + "_metadata.json")
            with open(outfile_fp, 'w') as of:
                json.dump(metadata, of, indent=2)
            print("Conservator metadata file saved at: {}".format(outfile_fp))

        elif format == "xml":
            """
            Save dataset in xml format
            Produces output_root/Annotations and
                     output_root/Data
            """
            image_save_dir = osp.join(output_root, 'Data')
            anno_save_dir = osp.join(output_root, 'Annotations')
            # Not thread safe
            if not osp.exists(image_save_dir):
                os.makedirs(image_save_dir)
            if not osp.exists(anno_save_dir):
                os.makedirs(anno_save_dir)

            for image in self.images:
                # save image file
                image.save_image(image_save_dir, flatten)
                # save annotation
                image.save_xml(anno_save_dir, flatten)

            print("Dataset saved as xml in {}".format(output_root))

        elif format == "detlog":
            """
            Save dataset in detlog format.
            Produces output_root/{dataset_name}.log
            Detection confidences will always be 0.999
            time1 and time2 will both always be 1
            """
            if not osp.exists(output_root):
                os.makedirs(output_root)

            detlog_name = self.name or "detlog"
            detlog_ext = ".txt"

            detlog_fp = os.path.join(output_root, detlog_name + detlog_ext)
            with open(detlog_fp, 'w') as detlog:
                for image in self.images:
                    for anno in image.annotations:
                        if not anno.cat_id:
                            raise Exception("To save in detlog format, you may need to provide a catids and labelvector")
                        lv_id = str(self.catid_to_lvid.get(anno.cat_id, "error"))
                        conf = 0.999
                        xmin = anno.l
                        ymin = anno.t
                        xmax = anno.r
                        ymax = anno.b
                        time1 = 1
                        time2 = 1
                        annotation_line = " ".join([str(bit) for bit in [image.image_filename,
                                                    lv_id,
                                                    conf,
                                                    xmin,
                                                    ymin,
                                                    xmax,
                                                    ymax,
                                                    time1,
                                                    time2]])
                        print(annotation_line)
                        detlog.write(annotation_line)
                        detlog.write("\n")
            print("detlog save to: {}".format(detlog_fp))

        elif format == "unified_coco":

            raise Exception("Unified coco save not yet implemented")

        else:
            raise Exception("Invalid save format")

    def load_detlog(self, logfile):
        """
        Read detections from a flir detection log file, produced by running (for example)
        nntc with flag `-log_detections`, or ssd_detect_FLIR.
        Each line has one bbox, formatted as:
            image filename, class id, conf, xmin, ymin, xmax, ymax, time1(optional), time2(optional)
        """
        images = defaultdict(list)
        with open(logfile, 'r') as detlog:
            for det in detlog:
                det_bits = det.split(' ')
                if det_bits[0] != 'LastRecord':
                    img_name = osp.splitext(osp.basename(det_bits[0]))[0]
                    #print img_name
                    images[img_name].append((det_bits))
        #print(images[images.keys()[0]])
        for image, detections in zip(images.keys(), images.values()):
            #print(detections[0][0])
            new_image = Image(image_name=image, image_filename=detections[0][0])
            new_image.ext = osp.splitext(new_image.image_filename)[1]
            im_w, im_h = pimage.open(new_image.image_filename).size
            new_image.width = im_w
            new_image.height = im_h
            for det in detections:
                c_id, __, x1, y1, x2, y2 = det[1:7]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                #print(c_id, x1, x2, y1, y2)
                new_anno = Annotation(cat_id=c_id, l=x1, r=x2, t=y1, b=y2)
                #new_anno.constrain_bbox(0, 0, im_w, im_h)
                #print(type(new_anno))
                new_image.add_anno(new_anno)

            self.images.append(new_image)

    def load_coco(self, src_dir, prefix_dir):
        """
        Load annotations from coco format.
        src_dir should contain /Annotations with .json annotation files, and
                               /Data wtih corresponding image files.
        """
        # Gather filenames with recursively with os.walk
        anno_fnames = []
        for root, dirnames, filenames in os.walk(osp.join(src_dir, 'Annotations')):
            for filename in fnmatch.filter(filenames, '*.json'):
                anno_fnames.append(os.path.join(root, filename))

        img_fnames = []
        img_search_dir = osp.join(src_dir, prefix_dir)
        for root, dirnames, filenames in os.walk(img_search_dir):
            for filename in fnmatch.filter(filenames, '*.*'):
                img_fnames.append(os.path.join(root, filename))

        img_fnames = sorted([img_fname for img_fname in img_fnames if osp.splitext(img_fname)[1] in ALLOWED_IMAGE_EXTENSIONS], key=natural_sort_key)
        anno_fnames = sorted(anno_fnames, key=natural_sort_key)
        if len(anno_fnames) != len(img_fnames):
            raise Exception("Number of annotation files != number of image files")

        img_count = 0
        for anno_fname, img_fname in zip(anno_fnames, img_fnames):
            new_image = Image(image_filename=img_fname, id=img_count)
            new_image.subdirs = osp.split(img_fname.replace(img_search_dir, './', 1))[0]

            with open(anno_fname, 'r') as af:
                im_anno_data = json.load(af)
                for annotation in im_anno_data['annotation']:
                    x, y, w, h = [int(bit) for bit in annotation['bbox']]
                    new_anno = Annotation(l=x, t=y, w=w, h=h)
                    new_anno.cat_id = annotation['category_id']
                    new_anno.id = annotation.get('id', None)
                    new_anno.is_crowd = annotation.get('iscrowd', None)
                    new_anno.segmentation = annotation.get('segmentation', None)
                    new_anno.image_id = annotation.get('image_id', None)
                    new_image.add_anno(new_anno)
            self.images.append(new_image)
            img_count += 1

    def load_xml(self, src_dir, prefix_dir):
        """
        Load annotations from xml format.
        This is the format conservator datasets download as.
        src_dir should contain /Annotations with .xml annotation files, and
                               /Data wtih corresponding image files.
        """
        # Gather filenames with recursively with os.walk
        anno_fnames = []
        for root, dirnames, filenames in os.walk(osp.join(src_dir, 'Annotations')):
            for filename in fnmatch.filter(filenames, '*.xml'):
                anno_fnames.append(os.path.join(root, filename))

        img_fnames = []
        img_search_dir = osp.join(src_dir, prefix_dir)
        for root, dirnames, filenames in os.walk(img_search_dir):
            for filename in fnmatch.filter(filenames, '*.*'):
                img_fnames.append(os.path.join(root, filename))

        img_fnames = sorted([img_fname for img_fname in img_fnames if osp.splitext(img_fname)[1] in ALLOWED_IMAGE_EXTENSIONS], key=natural_sort_key)
        anno_fnames = sorted(anno_fnames, key=natural_sort_key)
        if len(anno_fnames) != len(img_fnames):
            raise Exception("Number of annotation files != number of image files")

        img_count = 0
        for anno_fname, img_fname in zip(anno_fnames, img_fnames):
            print(anno_fname, " : ", img_fname)
            new_image = Image(image_filename=img_fname, id=img_count)
            new_image.subdirs = osp.split(img_fname.replace(img_search_dir, './', 1))[0]
            data_root = ET.parse(anno_fname).getroot()
            for anno in data_root.iter('object'):
                anno_name = anno.find('name')
                if anno_name is not None:
                    bbox = anno.find("bndbox")
                    cat_name = anno_name.text
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    new_anno = Annotation(l=xmin, t=ymin, r=xmax, b=ymax)
                    new_anno.name = cat_name
                    new_image.add_anno(new_anno)
                else:
                    print("Image ({}) has no annotations".format(anno_fname))

            self.images.append(new_image)
            img_count += 1

    def load_conservator(self, src_dir, annotation_file):
        """
        Load annotations from conservator metadata json format.
        annotation_file should be one .json file with annotations for all images.
        """
        img_fnames = []
        img_search_dir = osp.join(src_dir, prefix_dir)
        for root, dirnames, filenames in os.walk(img_search_dir):
            for filename in fnmatch.filter(filenames, '*.*'):
                img_fnames.append(os.path.join(root, filename))

        # Filter out non-images and sort by filename
        img_fnames = sorted([img for img in img_fnames if osp.splitext(img)[1] in ALLOWED_IMAGE_EXTENSIONS], key=natural_sort_key)
        for img_fname in img_fnames:
            self.images.append(Image(image_filename=img_fname))
            self.images.sort()

        if len(self.images) < 10:
            raise Exception("Found less than 10 images. Did you pass src_dir correctly?")

        with open(annotation_file, 'r') as data_file:
            metadata = json.load(data_file)
            if self.name is None:
                self.name = metadata["videos"][0]["name"]
            self.metadata = metadata["videos"][0]["description"]

            for new_image, frame_data in zip(self.images, metadata["videos"][0]["frames"]) :
                new_image.id = frame_data["frameIndex"]
                image_path = new_image.image_filename
                new_image.subdirs = osp.split(image_path.replace(img_search_dir, './', 1))[0]

                for anno_data in frame_data["annotations"]:
                    labels = anno_data["labels"]
                    x = anno_data["boundingBox"]["x"]
                    y = anno_data["boundingBox"]["y"]
                    w = anno_data["boundingBox"]["w"]
                    h = anno_data["boundingBox"]["h"]
                    new_anno = Annotation(l=x, t=y, w=w, h=h, name=labels[0])
                    new_image.add_anno(new_anno)

    def load_unified_coco(self, src_dir, annotation_file):
        with open(annotation_file, 'r') as data_file:
            data = json.load(data_file)
            self.metadata = data["info"]["description"]
            self.version = data["info"]["version"]
            self.date = data["info"]["date_created"]
            # Read images into id-mapped dict
            img_dict = {}
            for img in data["images"]:
                img_id = img["id"]
                fname = img["file_name"]
                full_filepath = osp.join(src_dir, fname)
                img_dict[img_id] = Image(image_filename=full_filepath, id=img_id, width=img["width"], height=img["height"])

            categories = data["categories"]
            cat_id_map = {}
            for cat in categories:
                cat_id_map[cat["id"]] = cat

            for anno in data["annotations"]:
                x, y, w, h = [round(coord) for coord in anno["bbox"]]
                new_anno = Annotation(l=x, t=y, w=w, h=h)
                new_anno.iscrowd = anno["iscrowd"]
                new_anno.cat_id = anno["category_id"]
                new_anno.segmentation = anno["segmentation"]
                new_anno.image_id = anno["image_id"]
                new_anno.id = anno["id"]
                new_anno.name = cat_id_map[new_anno.cat_id]["name"]
                new_anno.supercategory = cat_id_map[new_anno.cat_id]["supercategory"]
                img_dict[new_anno.image_id].add_anno(new_anno)

            for img_id, img in img_dict.items():
                self.images.append(img)

    def __str__(self):
        output_str = "Dataset Name: {} \n".format(self.name)
        output_str += "Number of images: {} \n".format(len(self.images))
        num_annos = sum([len(image.annotations) for image in self.images])
        output_str += "Number of annotations: {} \n".format(num_annos)
        output_str += "Metadata: {} \n".format(self.metadata)

        return output_str
