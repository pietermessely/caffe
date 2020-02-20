import os
import glob
import cv2
import json
from pprint import pprint
from shutil import copyfile

list = glob.glob('coco_2014/COCO/Annotations/train2014/*.json')

for json_file in list:

    data = json.load(open(json_file))

    reject = False

    for annotation in data["annotation"]:

        category_id = int(annotation["category_id"])

        if category_id == 1:      # person
            reject = True
        if category_id == 2:      # bicycle
            reject = True
        if category_id == 3:      # car
            reject = True
        if category_id == 4:      # motorcycle
            reject = True
        if category_id == 6:      # bus
            reject = True
        if category_id == 8:      # truck
            reject = True
        if category_id == 18:     # dog
            reject = True

    if not reject:

        # none of the above.  its a keeper.
        img_file = json_file.replace('.json', '.jpg')
        img_file = img_file.replace('/Annotations', '/images')

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("keeper", img)
        cv2.waitKey(1)

        #print(img_file)

        img_name = os.path.basename(img_file)
        json_name = os.path.basename(json_file)

        copyfile(json_file, '/mnt/data/HardNegative/HardNegativeAnnotation/' + json_name)
        copyfile(img_file, '/mnt/data/HardNegative/HardNegativeData/' + img_name)