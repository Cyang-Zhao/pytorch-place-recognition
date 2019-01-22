#!/anaconda/bin/env python
#!-*-coding:utf-8 -*-
#!@Time : 19-1-22 8:53
#!@Author : chenyangzhao@pku.edu.cn
#!@File : test.py

from network import *
from torchvision import transforms
from torch.utils.data import DataLoader
from data_generation import PairDataset
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((320, 240)),  # 缩放到 320 * 240 大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def default_loader(img_path):
    """
    :param img_path: the path of an image
    :return: the RGB image
    """
    return Image.open(img_path).convert('RGB')


def test(test_loader, net, select_layer):
    """

    :param test_loader: the data loader of test pairs
    :param net: the CNN used (feature extractor)
    :param select_layer: the convolutional layers that offer features
    :return: label and corresponding similarity of an test pair
    """
    print('Start to test!!!!!!')
    for i, pairs in enumerate(test_loader, 0):
        (img1, img2), (x1, x2), label = pairs
        if torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
        (out1_1, out1_2, out1_3, out1_4) = net.get_embedding(x1, select_layer)
        (out2_1, out2_2, out2_3, out2_4) = net.get_embedding(x2, select_layer)
        s_11 = cosine(out1_1, out2_1)
        s_22 = cosine(out1_2, out2_2)
        s_33 = cosine(out1_3, out2_3)
        s_44 = cosine(out1_4, out2_4)

        s_12 = cosine(out1_1, out2_2)
        s_13 = cosine(out1_1, out2_3)
        s_14 = cosine(out1_1, out2_4)
        s_23 = cosine(out1_2, out2_3)
        s_24 = cosine(out1_2, out2_4)
        s_34 = cosine(out1_3, out2_4)

        s_dia = average([s_11, s_22, s_33, s_44])
        s_off = average([s_12, s_13, s_14, s_23, s_24, s_34])
        sim = cal_similarity(s_dia, s_off)

        print('The similarity of {} and {} is {}, their label is {}'.format(img1[0], img2[0], sim[0], label[0]))


if __name__ == '__main__':
    # generate test data
    txt = './test_data/test.txt'
    test_path = './test_data/images'
    test_data = PairDataset(txt=txt,
                            test_path=test_path,
                            loader=default_loader,
                            transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

    # set model
    model = AlexNet()
    net = model.alexnet.features[0:11]
    my_model = TripletNet(net)
    print(my_model)
    # load weights
    my_model.embedding_net.load_state_dict(torch.load('./trained_params/Alexnet_params.pkl'))
    print('Load pkl success!!!!')
    if torch.cuda.is_available():
        print('CUDA OK!!!!')
        my_model = my_model.cuda()
        print('Convert to cuda success!!!!')
    # choose feature layers
    select_layer = [10]
    test(test_loader, my_model, select_layer)