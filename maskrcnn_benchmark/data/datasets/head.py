import os
import sys
import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

from maskrcnn_benchmark.structure.bounding_box import BoxList
from maskrcnn_benchmark.structure.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structure.segmentation_mask import Polygons

class HeadDataset(Dataset):
    def __init__(self, data_dir, gt_file, use_difficult=False,
            transform=None, is_train=True):
        self.data_dir = data_dir
        self.gt_file = os.path.join(data_dir, gt_file)
        if is_train:
            self.image_path_list, self.gt_list = \
                    self.get_annoatation(self.gt_file)
        else:
            self.image_path_list, _ = self.get_annoatation(self.gt_file)

        if transform is not None:
            self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_path = os.path.join(self.data_dir, image_path)
        image = Image.open(image_path)
        if self.gt_list is not None:
            gt = self.gt_list[idx]
            boxs = torch.tensor(gt['boxes'], dtype=torch.float32)
            boxlist =BoxList(boxs, image.size, mode='xyxy')
            labels = torch.tensor(len(gt['boxes']), dtype=torch.int)
            boxlist.add_field('labels', labels)

            masks = [(Polygons[poly]. image.size, "mask") for poly in gt["polys"]]
            masks = SegmentationMask(masks, image.size)
            boxlist.add_field('masks', masks)

            boxlist = boxlist.clip_to_image(remove_empty=True)
        else:
            pass

        return image, boxlist, idx

    def __len(self):
        return len(self.image_path_list)

    def get_annoatation(self, p):
        image_path_list = []
        gt_list = []
        lines = open(p).readlines()
        for line in lines:
            item = {}
            item['boxes'] = []
            item['ploys'] = []

            line = line.split(' ')
            image_path = line[0]
            item['path'] = image_path
            image_path_list.append(image_path)
            if line[1] == '0':
                # TODO: some image not contains head box,now don't konw how to deal with it
                continue
            line = list(map(float, line[1:]))
            head_nums = int(line[0])
            for i in range(head_nums):
                bbox = line[1+i*5, 6+i*5]
                bbox, poly = self.get_box_and_poly(bbox)
                item['boxes'].append(bbox)
                item['polys'].append(poly)
            gt_list.append(item)

        return image_path_list, gt_list

    def get_box_and_poly(self, bbox):
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y
        x3, y3 = x + w, y + h
        x4, y4 = x, y + h
        bbox = [x1, y1, x3, y3]
        poly = [x1, y1, x2, y2, x3, y3, x4, y4]
        return bbox, poly


def main():
    data_dir = "/workspace/csf/data/head_detect/train/yuncong_data/"
    gt_file = 'UCSD_train.txt'
    head = HeadDataset(data_dir, gt_file)
    from ipdb import set_trace
    set_trace()
    print('test')

if __name__ == '__main__':
    main()
