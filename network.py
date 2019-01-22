#!/anaconda/bin/env python
#!-*-coding:utf-8 -*-
#!@Time : 19-1-22 上午10:00
#!@Author : chenyangzhao@pku.edu.cn
#!@File : network.py

import torch
import torch.nn as nn
import torchvision.models as models


def globalAveragePooling(x):
    num = x.shape[2] * x.shape[3]
    x = x.sum(3)
    x = x.sum(2)
    x = x / num
    x = x.view(x.shape[0], -1)
    return x  # size:(batch, channel)


def cosine(x, y):
    """
    input: x,y
    output: x.dot(y)
    """
    x_length = x.pow(2).sum(1).pow(.5).view(-1,1)
    y_length = y.pow(2).sum(1).pow(.5).view(-1,1)
    x = x / x_length
    y = y / y_length
    return ((x * y).sum(1)+1)/2


def average(x):
    """
    input: similarities
    output: sum()/4 or sum()/6
    """
    x_sum = x[0]
    for i in range(1, len(x)):
        x_sum = x_sum + x[i]
    return x_sum/len(x)


def cal_similarity(value_dia, value_off):
    """
    input: similarities of diagonal values and off-diagonal values
    output: alpha*value_dia
    """
    d = (value_off-value_dia)
    e = torch.exp(15*d+1.5)
    alpha = 1/(1+e)
    return alpha*value_dia


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=False)

    def forward(self, x):
        output = self.alexnet(x)
        return output

    def get_embedding(self, x):
        return self.alexnet(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        an_1 = x1[:, :, 0:120, 0:160]
        an_2 = x1[:, :, 0:120, 160:320]
        an_3 = x1[:, :, 120:240, 0:160]
        an_4 = x1[:, :, 120:240, 160:320]

        pos_1 = x2[:, :, 0:120, 0:160]
        pos_2 = x2[:, :, 0:120, 160:320]
        pos_3 = x2[:, :, 120:240, 0:160]
        pos_4 = x2[:, :, 120:240, 160:320]

        neg_1 = x3[:, :, 0:120, 0:160]
        neg_2 = x3[:, :, 0:120, 160:320]
        neg_3 = x3[:, :, 120:240, 0:160]
        neg_4 = x3[:, :, 120:240, 160:320]

        out_an1 = globalAveragePooling(self.embedding_net(an_1))
        out_an2 = globalAveragePooling(self.embedding_net(an_2))
        out_an3 = globalAveragePooling(self.embedding_net(an_3))
        out_an4 = globalAveragePooling(self.embedding_net(an_4))

        out_po1 = globalAveragePooling(self.embedding_net(pos_1))
        out_po2 = globalAveragePooling(self.embedding_net(pos_2))
        out_po3 = globalAveragePooling(self.embedding_net(pos_3))
        out_po4 = globalAveragePooling(self.embedding_net(pos_4))

        out_ne1 = globalAveragePooling(self.embedding_net(neg_1))
        out_ne2 = globalAveragePooling(self.embedding_net(neg_2))
        out_ne3 = globalAveragePooling(self.embedding_net(neg_3))
        out_ne4 = globalAveragePooling(self.embedding_net(neg_4))

        # similarity between anchor and positive
        ap_11 = cosine(out_an1, out_po1)
        ap_22 = cosine(out_an2, out_po2)
        ap_33 = cosine(out_an3, out_po3)
        ap_44 = cosine(out_an4, out_po4)

        ap_12 = cosine(out_an1, out_po2)
        ap_13 = cosine(out_an1, out_po3)
        ap_14 = cosine(out_an1, out_po4)
        ap_23 = cosine(out_an2, out_po3)
        ap_24 = cosine(out_an2, out_po4)
        ap_34 = cosine(out_an3, out_po4)

        ap_dia = average([ap_11, ap_22, ap_33, ap_44])
        ap_off = average([ap_12, ap_13, ap_14, ap_23, ap_24, ap_34])
        similarity_ap = cal_similarity(ap_dia, ap_off)

        # similarity between anchor and negative
        an_11 = cosine(out_an1, out_ne1)
        an_22 = cosine(out_an2, out_ne2)
        an_33 = cosine(out_an3, out_ne3)
        an_44 = cosine(out_an4, out_ne4)

        an_12 = cosine(out_an1, out_ne2)
        an_13 = cosine(out_an1, out_ne3)
        an_14 = cosine(out_an1, out_ne4)
        an_23 = cosine(out_an2, out_ne3)
        an_24 = cosine(out_an2, out_ne4)
        an_34 = cosine(out_an3, out_ne4)

        an_dia = average([an_11, an_22, an_33, an_44])
        an_off = average([an_12, an_13, an_14, an_23, an_24, an_34])
        similarity_an = cal_similarity(an_dia, an_off)

        print('similarity_ap:', similarity_ap)
        print('similarity_an:', similarity_an)

        return similarity_ap, similarity_an

    def get_embedding(self, x, select_layer):
        # extract features from specific layer and then concatenate them.
        part1 = x[:, :, 0:120, 0:160]
        part2 = x[:, :, 0:120, 160:320]
        part3 = x[:, :, 120:240, 0:160]
        part4 = x[:, :, 120:240, 160:320]

        input = [part1, part2, part3, part4]
        out = [[],[],[],[]]
        for i, part in enumerate(input):
            for index, module in enumerate(self.embedding_net):
                part = module(part)
                if index in select_layer:
                    out[i].append(part)
        out_final = [[], [], [], []]
        for i, part_out in enumerate(out):
            for j, out_layer in enumerate(part_out):
                out_layer = globalAveragePooling(out_layer)
                if j == 0:
                    out_final[i].append(out_layer)
                else:
                    out_final[i][0] = torch.cat([out_final[i][0], out_layer], 1)

        output = (out_final[0][0], out_final[1][0], out_final[2][0], out_final[3][0])
        return output