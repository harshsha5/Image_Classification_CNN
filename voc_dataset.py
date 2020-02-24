# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET
from skimage import io, transform
from scipy import misc

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset
import ipdb


class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    def __init__(self, split, size, data_dir='../VOCdevkit_Toy/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]     #index_list is stores the index or basically the names of the images
        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of lables. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        label_list = []
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()
            # TODO: insert your code here, preload labels
            labels = np.zeros(len(VOCDataset.CLASS_NAMES))
            weights = np.ones(len(VOCDataset.CLASS_NAMES))
            for obj in root.findall('object'):
                naam = obj.find('name').text
                difficulty = obj.find('difficult').text
                labels[VOCDataset.get_class_index(naam)] = 1
                if(difficulty=='0'):
                    weights[VOCDataset.get_class_index(naam)] = np.inf
                else:
                    weights[VOCDataset.get_class_index(naam)] -= 1
            weights = np.clip(weights, 0, 1)
            label_list.append([labels,weights])            

        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]
        fpath = os.path.join(self.img_dir, findex + '.jpg')
        newsize = (256,256) 
        # TODO: insert your code here. hint: read image, find the labels and weight.
        img = io.imread(fpath)
        img = misc.imresize(img,newsize)
        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]
        img = torch.FloatTensor(img)
        image = img.permute(2, 0, 1)
        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)

        return image, label, wgt

