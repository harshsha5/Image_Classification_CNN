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
    for i in range(1,len(fc7_output)):
        intermediate_output = torch.cat([intermediate_output, torch.squeeze(fc7_output[i])], dim=0)

    #Add t-SNE code here

if __name__ == '__main__':
    args, device = utils.parse_args()
    main()