import argparse
import xml.etree.ElementTree as ElTr
import os

parser = argparse.ArgumentParser(description='Build Annotations.')
parser.add_argument('dir', default='..', help='Annotations.')

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}


def convert_annotation(annotation_dir, year, image_id, f):
    in_file = os.path.join(annotation_dir, 'VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ElTr.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        classes = list(classes_num.keys())
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text),
             int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))


def _main(args):
    data_dir = os.path.expanduser(args.dir)

    for year, image_set in sets:
        with open(os.path.join(data_dir, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), 'r') as f:
            image_ids = f.read().strip().split()
        with open(os.path.join(data_dir, '%s_%s.txt' % (year, image_set)), 'w') as f:
            for image_id in image_ids:
                f.write('%s/VOC%s/JPEGImages/%s.jpg' % (data_dir, year, image_id))
                convert_annotation(data_dir, year, image_id, f)
                f.write('\n')


if __name__ == '__main__':
    # _main(parser.parse_args())
    # _main(parser.parse_args(['D:/Datasets/VOC/VOCdevkit']))
    _main(parser.parse_args(['C:\\BaiduNetdiskDownload\\pascalvoc\\VOCdevkit']))