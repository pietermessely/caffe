from __future__ import print_function

from collections import OrderedDict
from annotation import Annotation
from PIL import Image as pimage
import xml.etree.ElementTree as ET
import os.path as osp
import shutil
import json
import re
import os


ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.png', '.tiff', '.jpeg', '.gif', '.bmp']
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

class Image(object):
    def __init__(self, **kwargs):
        self.image_filename = kwargs.get('image_filename', None)
        self.subdirs = kwargs.get('subdirs', None) # Directories between src_dir/prefix_dir and image file
        self.image_name = kwargs.get('name', None)
        self.ext = kwargs.get('ext', None)
        self.id = kwargs.get('id', None)
        self.image = kwargs.get('image', None)
        self.width = kwargs.get('width', None)
        self.height = kwargs.get('height', None)
        self.depth = kwargs.get('depth', 3)
        self.annotations = kwargs.get('annos', [])

        if not self.image_filename:
            raise Exception("Image needs to have a filename")
        if not self.image_name:
            self.image_name = osp.splitext(osp.basename(self.image_filename))[0]
        if not self.ext:
            self.ext = osp.splitext(self.image_filename)[1]
            self.ext = self.ext.lower()
        if not self.width or not self.height:
            self.width, self.height = pimage.open(self.image_filename).size

    def __str__(self):
        return "{} {}x{}".format(self.image_filename, self.width, self.height)
    def __gt__(self, other):
        return natural_sort_key(self.image_name) > natural_sort_key(other.image_name)
    def __lt__(self, other):
        return natural_sort_key(self.image_name) < natural_sort_key(other.image_name)
    def __eq__(self, other):
        return natural_sort_key(self.image_name) == natural_sort_key(other.image_name)

    def add_anno(self, annotation):
        """
        Add an annotation to the image
        """
        if isinstance(annotation, Annotation):
            self.annotations.append(annotation)
        else:
            raise Exception("Did not pass annotation")

    def filter_catids(self, catids):
        """
        Remove all annotations not in catids
        """
        catids = [int(id) for id in catids]
        self.annotations = [annotation for annotation in self.annotations if annotation.cat_id not in catids]

    def remap_catids(self, mapping):
        """
        Remap all annotation catids (Annotation.cat_id)
        [string(int) -> int]
        """
        for anno in self.annotations:
            anno.cat_id = mapping.get(str(anno.cat_id), anno.cat_id)

    def remap_labels(self, mapping):
        """
        Remap all annotation labels (Annotation.name)
        [string -> string]
        """
        for anno in self.annotations:
            #print(anno.name, mapping.get(str(anno.name), "butt"))
            #b4 = anno.name
            anno.name = mapping.get(str(anno.name).lower(), anno.name)
            #if b4 == anno.name:
            #    print("{} : {}".format(b4, anno.name))

    def sanitize(self, catids_map=None, catid_filter=None, catid_to_name=None):
        """
        Check if: - filename exists
                  - file extension is an image extension
                  - image width & height are non-zero
                  - Constrains annotations to image boundary
                  - Filter out catids not in catid_filter
        Then sanitize all annotations in the image
        """
        if not osp.isfile(self.image_filename):
            raise Exception("Invalid filename: {}".format(self.image_filename))
        if self.ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise Exception("File extension not recognized ({})".format(self.ext))
        if self.width == 0 or self.height == 0:
            raise Exception("Invalid width or height for image: {}".format(self.image_filename))

        #print("sanitizing: {}".format(self.image_filename))
        self.annotations = [anno for anno in self.annotations if anno.w > 0 and anno.h > 0]
        for anno in self.annotations:
            anno.constrain_bbox(0, 0, self.width, self.height)
            anno.image_id = self.id
            anno.sanitize(catids_map=catids_map, catid_to_name=catid_to_name)

        if catid_filter:
            self.filter_catids(catid_filter)

        if self.id is None:
            self.id = abs(hash(self.image_name)) % (10 ** 8)

        if self.subdirs is None:
            self.subdirs = "./"

    def save_image(self, location, flatten=True):
        """
        Save this image to location
        """
        if self.image_filename:
            try:
                if flatten:
                    shutil.copyfile(self.image_filename, osp.join(location, self.image_name + self.ext))
                else:
                    save_dir = osp.join(location, self.subdirs)
                    if not osp.exists(save_dir):
                        os.makedirs(save_dir)
                    shutil.copyfile(self.image_filename, osp.join(save_dir, self.image_name + self.ext))
            except Exception as err:
                print(err)
                print("unable to copy image: ", self.image_filename)
                print("To: {}".format(osp.join(location, self.subdirs, self.image_name + self.ext)))
        else:
            raise Exception("Need original image filename to be able to copy image")

    def save_coco(self, location, flatten=True):
        """
        Save this image's annotations in coco format
        """
        outfile = ""
        if flatten:
            outfile = osp.join(location, self.image_name + '.json')
        else:
            save_dir = osp.join(location, self.subdirs)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            outfile = osp.join(save_dir, self.image_name + '.json')

        out_json = OrderedDict([
            ('image', OrderedDict([
                ('file_name', self.image_name + self.ext),
                ('id', self.id),
                ('width', self.width),
                ('height', self.height)
            ])),
            ('annotation', [])
        ])

        for anno in self.annotations:
            anno_json = OrderedDict()
            if anno.area is not None:
                anno_json['area'] = anno.area
            anno_json['bbox'] = [anno.l, anno.t, anno.w, anno.h]
            if anno.cat_id is not None:
                anno_json['category_id'] = anno.cat_id
            else:
                print("WARNING: No category_id for annotation in image {}. Passing a catids file might fix.".format(self.image_filename))
            if anno.id is not None:
                anno_json['id'] = anno.id
            if anno.image_id is not None:
                anno_json['image_id'] = anno.image_id
            anno_json['iscrowd'] = anno.is_crowd
            if anno.segmentation is not None:
                anno_json['segmentation'] = anno.segmentation

            out_json['annotation'].append(anno_json)
        with open(outfile, 'w') as of:
            json.dump(out_json, of, indent=2)

    def save_xml(self, location, flatten=True):
        """
        Save this image's annotations in xml format
        """
        outfile = ""
        if flatten:
            outfile = osp.join(location, self.image_name + '.xml')
        else:
            save_dir = osp.join(location, self.subdirs)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            outfile = osp.join(save_dir, self.image_name + '.xml')

        anno_root = ET.Element('annotation')
        segmented = ET.SubElement(anno_root, 'segmented')
        segmented.text = '0'
        for anno in self.annotations:
            anno_obj = ET.SubElement(anno_root, 'object')
            anno_bbox = ET.SubElement(anno_obj, 'bndbox')
            xmin = ET.SubElement(anno_bbox, 'xmin')
            ymin = ET.SubElement(anno_bbox, 'ymin')
            xmax = ET.SubElement(anno_bbox, 'xmax')
            ymax = ET.SubElement(anno_bbox, 'ymax')
            xmin.text = str(anno.l)
            ymin.text = str(anno.t)
            xmax.text = str(anno.r)
            ymax.text = str(anno.b)

            difficult = ET.SubElement(anno_obj, 'difficult')
            difficult.text = '0'
            pose = ET.SubElement(anno_obj, 'pose')
            pose.text = "Unspecified"
            name = ET.SubElement(anno_obj, 'name')
            if not anno.name:
                raise Exception("Anno has no name. Probably needs a catids/labelvector.")
            name.text = anno.name
            truncated = ET.SubElement(anno_obj, 'truncated')
            truncated.text = "0"

        fname = ET.SubElement(anno_root, "filename")
        fname.text = self.image_name + self.ext

        source = ET.SubElement(anno_root, "source")
        database = ET.SubElement(source, "database")
        database.text = 'Flir database'

        folder = ET.SubElement(anno_root, 'folder')
        folder.text = location

        size = ET.SubElement(anno_root, "size")
        xml_width = ET.SubElement(size, "width")
        xml_width.text = str(self.width)
        xml_height = ET.SubElement(size, "height")
        xml_height.text = str(self.height)
        xml_depth = ET.SubElement(size, "depth")
        xml_depth.text = str(self.depth)

        xml_stringified = ET.tostring(anno_root)
        open(outfile, 'w').write(xml_stringified)
