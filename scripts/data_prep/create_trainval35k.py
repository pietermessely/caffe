import shutil

val_minus_minival = open('/mnt/data/flir-data/coco/COCO/ImageSets/valminusminival2014.txt', 'r')
train_val = open('/mnt/data/flir-data/coco/COCO/ImageSets/train2014.txt', 'r')

for line in val_minus_minival:
    line = line.strip()
    src_anno  = '/mnt/data/coco_2014/COCO/Annotations/valminusminival2014/{}.json'.format(line)
    src_image = '/mnt/data/flir-data/coco/COCO/images/val2014/{}.jpg'.format(line)

    dst_anno  = '/mnt/data/coco_2014/COCO/trainval35k/Annotations/{}.json'.format(line)
    dst_image = '/mnt/data/coco_2014/COCO/trainval35k/Data/{}.jpg'.format(line)
    print(src_image)
    try:
        shutil.copyfile(src_anno, dst_anno)
        shutil.copyfile(src_image, dst_image)
    except:
        print('failed:' +  src_image) 
        continue
 
for line in train_val:
    line = line.strip()
    src_anno  = '/mnt/data/coco_2014/COCO/Annotations/train2014/{}.json'.format(line)
    src_image = '/mnt/data/flir-data/coco/COCO/images/train2014/{}.jpg'.format(line)

    dst_anno  = '/mnt/data/coco_2014/COCO/trainval35k/Annotations/{}.json'.format(line)
    dst_image = '/mnt/data/coco_2014/COCO/trainval35k/Data/{}.jpg'.format(line)
    print(src_image)
    try:
       	shutil.copyfile(src_anno, dst_anno)
        shutil.copyfile(src_image, dst_image)
    except:
        print('failed:' +  src_image) 
        continue
