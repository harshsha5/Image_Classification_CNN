# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse

import numpy as np
import os
import sklearn.metrics

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import ipdb

def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='Assignment 1')
    # config for dataset

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: .001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before evaluating model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model_to_use', type=int, default=1, metavar='NI',            #1 for SimpleCNN #2 for CaffeNet #3 for Resnet
                        help='What model to use')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


def get_data_loader(name='voc', train=True, batch_size=64, split='train', model_to_use=1):
    if name == 'voc':
        from voc_dataset import VOCDataset
        dataset = VOCDataset(split, 64, model_to_use)
    else:
        raise NotImplementedError

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
    )
    return loader


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, device, test_loader):
    """
    Evaluate the model with the given dataset
    Args:
         model (Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    with torch.no_grad():
        count = 0
        total_test_loss = 0
        for data, target, wgt in test_loader:
            ## TODO insert your code here
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)
            output = model(data)
            total_test_loss += F.multilabel_soft_margin_loss(output,wgt*target)
            if(count==0):
                gt = np.clip((wgt*target).cpu().clone().numpy(), 0, 1)
                pred = torch.sigmoid(output)
                #Use sigmoid here, as each output tensor would be a logit whose probability we need. The probability might
                                              #not sum to 1 as this is multi-label classification and each probab is independent of other classes
                pred = pred.cpu().clone().numpy()
                valid = wgt.cpu().clone().numpy()
            else:
                gt = np.vstack((gt,np.clip((wgt*target).cpu().clone().numpy(), 0, 1)))
                pred = np.vstack((pred,torch.sigmoid(output).cpu().clone().numpy()))
                valid = np.vstack((valid,wgt.cpu().clone().numpy()))

            count+=1

    average_test_loss_per_iteration = total_test_loss/count
    AP = compute_ap(gt, pred, valid)

    mAP = np.mean(AP)
    return AP, mAP, average_test_loss_per_iteration

