# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
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

def visualize_data_sample(tensor_img):
    tensor_img = tensor_img.int()
    plt.imshow(tensor_img.permute(1, 2, 0))
    plt.show()

def visualize_filter(kernels,epoch,path):
    #Referenced from https://discuss.pytorch.org/t/visualize-feature-map/29597/6
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    img = make_grid(kernels)
    plt.imshow(img.permute(1, 2, 0).cpu().clone())
    plt.savefig(path+'_'+str(epoch))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

fc7_output = []
def hook(module, input, output2):
    fc7_output.append(output2)

def get_all_indices(dicto):
    '''
    Extracts all unique indices from the Dictionary
    Input:
    dicto: Dictionary containing all the keys(as ints) and the values(as tensors). The keys here represent a sample image and the values represent it's nearest neighbors
    Output:
    A list of all unique elements in the dictionary
    '''
    x = set()
    for key,value in dicto.items():
        x.add(key)
        for val in value:
            x.add(val.item())
    return list(x)

def get_nearest_neighbors(num_images,num_neighbors,intermediate_output):
    '''
    num_images: Number of images whose neighbors you wish to compute
    num_neighbors: Number of neighbors you wish to compute
    intermediate_output(N X layer_length Tensor): The intermediate tensor output from the network

    Returns: Dictionary mapping the image with it's nearest neighbors
    '''
    dicto = {}
    l2_dist = nn.PairwiseDistance(p=2)
    for i in range(num_images):
        normed_distance = l2_dist(intermediate_output, intermediate_output[i].view(1,-1).repeat(intermediate_output.shape[0],1))
        res, ind = torch.topk(normed_distance,num_neighbors+1,largest=False)
        ind = ind[1:]      #Removing self as the closest neighbor
        dicto[i] = ind
    return dicto

def main():
    # TODO:  Initialize your visualizer here!
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

    # TODO: complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval',model_to_use = int(args.model_to_use))
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test',model_to_use = int(args.model_to_use))

    dataiter = iter(train_loader)
    for i in range(10):
        images, labels, wgt = dataiter.next()
        img_grid = make_grid(images)
        plt.imshow(img_grid.permute(1, 2, 0).cpu().clone())
        plt.show()

    # 2. define the model, and optimizer.
    # TODO: modify your model here!
    # bad idea of use simple CNN, but let's give it a shot!
    # In task 2, 3, 4, you might want to modify this line to be configurable to other models.
    # Remember: always reuse your code wisely.
    if(int(args.model_to_use)==1):
        model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3).to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_simpleCNNmodel"
        print("Using SimpleCNN")
    elif(int(args.model_to_use)==2):
        model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3,dropout_prob=0.5).to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_CaffeNet"

        '''Uncomment for T-SNE'''
        # model.fc2.register_forward_hook(hook)

        print("Using CaffeNet")
    elif(int(args.model_to_use)==3):
        model = models.resnet18(pretrained=False)
        final_layer_in_features = model.fc.in_features
        model.fc = nn.Linear(final_layer_in_features, len(VOCDataset.CLASS_NAMES))
        model = model.to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_Resnet_no_pretrain"
        print("Using ResNet")
    elif(int(args.model_to_use)==4):
        model = models.resnet18(pretrained=True)
        final_layer_in_features = model.fc.in_features
        model.fc = nn.Linear(final_layer_in_features, len(VOCDataset.CLASS_NAMES))
        model.fc.weight = nn.Parameter(2*torch.rand(model.fc.weight.shape)-1)   #This ensures that the last layer gets randomly allotted weights in the range [-1,1]
        model = model.to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_Resnet_pretrained"
        print("Using ResNet")
    else:
        print("Select Correct model_to_use")
        raise NotImplementedError

    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.01,patience = 2,verbose=True)
    cnt = 0

    #Model Save Code        #Get 5 checkpoints
    model_save_epochs = np.linspace(0,args.epochs-1, num=5, endpoint=True,dtype=int)
    index_model_save_epochs = 0

    #Visualize Filter Code          #Visualize at 3 different epochs
    filter_visualization_epochs = np.linspace(0,args.epochs-1, num=3, endpoint=True,dtype=int)
    index_filter_visualization_epochs = 0    

    for epoch in range(args.epochs):
        print("---------------EPOCH: ",epoch," ------------------------")
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)
            # visualize_data_sample(data[0])
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            loss = F.multilabel_soft_margin_loss(output,wgt*target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Train/Loss', loss, cnt)
                current_lr = get_lr(copy.deepcopy(optimizer))
                writer.add_scalar('lr', current_lr, cnt)
                if(args.model_to_use==3 or args.model_to_use==4):
                    writer.add_histogram('Resnet_Conv1', model.conv1.weight.grad, cnt)
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map, avg_test_loss = utils.eval_dataset_map(model, device, test_loader)
                writer.add_scalar('MAP', map, cnt)
                writer.add_scalar('Average_Test_Loss', avg_test_loss, cnt)
                scheduler.step(avg_test_loss)
                model.train()
            cnt += 1
        #scheduler.step()

        if(model_save_epochs[index_model_save_epochs] == epoch):
            print("Saving Model ",epoch)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, MODEL_SAVE_PATH)
            index_model_save_epochs+=1

        if(filter_visualization_epochs[index_filter_visualization_epochs] == epoch):        #Filters are only needed for CaffeNet so can be commented for Resnet and SimpleCNN
            print("Extracting Filter",epoch)
            index_filter_visualization_epochs+=1
            kernels = model.conv1.weight.detach().clone()
            visualize_filter(kernels,epoch,MODEL_SAVE_PATH)
            writer.add_image('Train_Images_'+str(epoch)+'_1', data[0])            #Uncomment for ResNet Question

    if(int(args.model_to_use)==2):
        model.fc2.register_forward_hook(hook)
        print("Registered hook to model.fc2 for Caffe_Net")     #Change this hook appropriately based on the question
    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test',model_to_use = int(args.model_to_use))
    ap, map, avg_test_loss= utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    print(ap)
    print('mAP: ', map)
    print('Average_Test_Loss: ', avg_test_loss)

    '''TODO: For T-SNE part'''
    # if(int(args.model_to_use)==2):
    #     tsne = manifold.TSNE(n_components=2, init='random', perplexity=10)
    #     trans_data = tsne.fit_transform(fc7_output[-1])
    #     plt.figure()
    #     ax = plt.subplot(111)
    #     plt.scatter(trans_data[0], trans_data[1], cmap=plt.cm.rainbow)
    #     plt.show()

    '''Nearest Neighbor Analysis '''
    if(int(args.model_to_use)==2):
        num_images = 3
        num_neighbors = 3
        intermediate_output = fc7_output[0]
        for i in range(1,len(fc7_output)):
            intermediate_output = torch.cat([intermediate_output, fc7_output[i]], dim=0)
        dicto = get_nearest_neighbors(num_images,num_neighbors,intermediate_output)
        sample_index_list = get_all_indices(dicto)

        with torch.no_grad():
            batch_count = 1
            image_index_mapping = {}
            for data, target, wgt in test_loader:
                feasible_indices = [i for i in sample_index_list if (i < int(args.test_batch_size)*batch_count and i>=int(args.test_batch_size)*(batch_count-1))]
                feasible_indices = [x - int(args.test_batch_size)*(batch_count-1) for x in feasible_indices]
                for index in feasible_indices:
                    image_index_mapping[index] = data[index] 
                batch_count+=1

            counter = 0
            for key,val in dicto.items():
                nearest_neighbors = torch.unsqueeze(image_index_mapping[key],0)    #See Size!
                for elt in val:
                    nearest_neighbors = torch.cat([nearest_neighbors, torch.unsqueeze(image_index_mapping[elt.item()],0)], dim=0)
                img_grid = make_grid(nearest_neighbors)
                plt.imshow(img_grid.permute(1, 2, 0).cpu().clone())
                plt.show()
                writer.add_image('Nearest_Neighbors_for_image'+str(counter), img_grid)
    writer.close()



if __name__ == '__main__':
    args, device = utils.parse_args()
    main()
