#!/anaconda/bin/env python
#!-*-coding:utf-8 -*-
#!@Time : 19-1-22 上午10:04
#!@Author : chenyangzhao@pku.edu.cn
#!@File : data_generation.py

import os
from torch.utils.data import Dataset, DataLoader


# load exit pairs in txt
class PairDataset(Dataset):
    """
    txt: test.txt
    test_path: the path of images
    return: names of images, images and label
    """
    def __init__(self, txt, test_path, loader, transform=None, ):
        fh = open(txt, 'r')
        imgs1 = []
        imgs2 = []
        label = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs1.append(words[0])
            imgs2.append(words[1])
            label.append(int(words[2]))

        self.imgs1 = imgs1
        self.imgs2 = imgs2
        self.label = label
        self.test_path = test_path
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn1 = self.imgs1[index]
        fn2 = self.imgs2[index]
        label = self.label[index]
        img_path1 = os.path.join(self.test_path, fn1)
        img_path2 = os.path.join(self.test_path, fn2)
        img1 = self.loader(img_path1)
        img2 = self.loader(img_path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (fn1, fn2), (img1, img2), label

    def __len__(self):
        return len(self.imgs1)