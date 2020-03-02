from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import utils
from q0_hello_mnist import SimpleCNN
from Caffe_Net import CaffeNet
from voc_dataset import VOCDataset
import ipdb

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import time
from torchvision.utils import make_grid
from torchvision import models
import copy
from sklearn import manifold

fc7_output = []
def hook(module, input, output2):
    fc7_output.append(output2)

def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

    # train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval',model_to_use = int(args.model_to_use))
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test',model_to_use = int(args.model_to_use))

    MODEL_PATH = "../Saved_Models/trained_CaffeNet"
    model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3,dropout_prob=0.5).to(device)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.fc2.register_forward_hook(hook)
    ap, map, avg_test_loss= utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    print(ap)
    print('mAP: ', map)
    print('Average_Test_Loss: ', avg_test_loss)

    intermediate_output = torch.squeeze(fc7_output[0])
    NUM_IMAGES_TO_MAKE_TSNE = 1000
    assert(int(args.test_batch_size)==NUM_IMAGES_TO_MAKE_TSNE)
    # ipdb.set_trace()
    dataiter = iter(test_loader)
    images,labels,wgt = dataiter.next()

    assert(labels.shape[0]==1000)   #Make sure that the test batch size is 1000

    #Add t-SNE code here
    intermediate_output = intermediate_output.cpu().numpy()
    intermediate_output_embedded = manifold.TSNE(n_components=2).fit_transform(intermediate_output)

    # colors = ["black","grey","lightcoral","red","saddlebrown","darkorange","gold","olivedrab","lawngreen","turquoise",
    #             "deepskyblue","royalblue","blue","indigo","magenta","crimson","orangered","goldenrod","steelblue","springgreen"]
    colors = [(0.3,0.3,0.5),(0.1,0.2,0.3),(0.2,0.1,0.5),(0.1,0.6,0.6),(0.9,0.3,0.1),(0.8,0.2,0.2),(0.4,0.4,0.4),(0.1,0.7,0.6),(0.2,0.8,0.5),(0.1,0.6,0.1),
                (0,0.1,0.4),(0.5,0.5,0),(0.6,0,0.5),(0.2,0.4,0.9),(0,0,0.9),(0.2,0.3,0.6),(0.9,0.5,0.5),(0.7,0.8,0.9),(0.2,0.1,0.1),(0.9,0.5,0.1)]

    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    for row in range(intermediate_output_embedded.shape[0]):
        im_label_vector = labels[row].cpu().numpy()
        positive_labels = np.argwhere(im_label_vector==1).flatten()

        total_r,total_g,total_b = 0,0,0
        for i,elt in enumerate(positive_labels):
            total_r+= colors[positive_labels[i]][0]
            total_g+= colors[positive_labels[i]][1]
            total_b+= colors[positive_labels[i]][2]

        i+=1   
        avg_r,avg_g,avg_b = total_r/i, total_g/i, total_b/i
        color_array = np.array([[avg_r,avg_g,avg_b ]])
        # ipdb.set_trace()
        # plt.scatter(intermediate_output_embedded[0], intermediate_output_embedded[1], c=(avg_r,avg_g,avg_b))
        if(i==1):
            plt.scatter(intermediate_output_embedded[row][0], intermediate_output_embedded[row][1], c=color_array,label = CLASS_NAMES[positive_labels[0]])
        else:
            plt.scatter(intermediate_output_embedded[row][0], intermediate_output_embedded[row][1], c=color_array)
        plt.xlabel("Dim_1")
        plt.ylabel("Dim_2")
        plt.legend(loc='upper left')
    # plt.legend((lo, ll, l, a, h, hh, ho),
    #    ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
    #    scatterpoints=1,loc='upper right',title="Classes",ncol=4,fontsize=8)
    plt.savefig("t-sne visualization")
    print("t-SNE visualization saved")
    plt.show()


if __name__ == '__main__':
    args, device = utils.parse_args()
    main()
