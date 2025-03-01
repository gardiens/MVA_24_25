
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
        
class RandomScale(object):
    def __call__(self, pointcloud):
        scale = np.random.rand()+0.5 
        scaled_pointcloud = pointcloud * scale
        return scaled_pointcloud
        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])
def default_transforms2():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),RandomScale(),ToTensor()])
def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        # YOUR CODE
        super(MLP,self).__init__()
        self.fc1=nn.Linear(3072,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,classes)
        self.dropout=nn.Dropout(0.3)
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(256)
        self.acti=nn.ReLU()
        

    def forward(self, input):
        # YOUR CODE

        x=input.flatten(start_dim=1)
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.acti(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.acti(x)
        x=self.dropout(x)
        x=self.fc3(x)
        return x



class MLP_con1D_Trick(nn.Module):
    # Trick that will used to apply MLP on Point Clouds 
    def __init__(
        self,
        inp_dim=3,
        hidden_dims1=[64, 128, 1024],
        use_bn=True,
        activation=nn.ReLU,
        last_layer_acti=True,
    ):
        super().__init__()
        layers_list = []
        self.inp_dim = inp_dim

        if self.inp_dim != hidden_dims1[0]:
            layers_list.append(
                torch.nn.Conv1d(
                    in_channels=self.inp_dim,
                    out_channels=hidden_dims1[0],
                    kernel_size=1,
                )
            )
            layers_list.append(activation())
        for k in range(len(hidden_dims1) - 1):
            if use_bn:
                layers_list.append(nn.BatchNorm1d(num_features=hidden_dims1[k]))

            layers_list.append(
                torch.nn.Conv1d(
                    in_channels=hidden_dims1[k],
                    out_channels=hidden_dims1[k + 1],
                    kernel_size=1,
                )
            )

            if k == len(hidden_dims1) - 2:
                if last_layer_acti:
                    layers_list.append(activation())
            else:
                layers_list.append(activation())

        self.first_mlp = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.first_mlp(x)
    





class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        #* A 2 layers shared MLP over points to compute features of dim 64: ( 3,64) then 64,64
        self.first_mlp=MLP_con1D_Trick(
            inp_dim=3,
            hidden_dims1=[64,64],
            use_bn=True,
            activation=nn.ReLU
        )

        #* A 3 layers shared MLP over points to compute features of dim 1024 (64,128,1024)
        self.second_mlp=MLP_con1D_Trick(
            inp_dim=64,
            hidden_dims1=[64,64,128,1024],

            use_bn=True,
            activation=nn.ReLU
        )


        # finally a 3 layers MLP to compute the classification scores
        # YOUR CODE
        self.last_mlp=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,classes)
        )

    def forward(self, input):
        # input=input.transpose(2,1) # put input in the convolution foramt
        x=self.first_mlp(input)
        x=self.second_mlp(x)
        # apply max_pooling as global feature vecotr

        global_feature = torch.max(x, dim=-1).values
        out=self.last_mlp(global_feature)

        return out 
        # YOUR CODE

        
        

class Tnet(nn.Module):
    def __init__(
        self,
        inp_dim=3,
        hidden_dims1=[64, 128, 1024],
        hidden_dims2=[512, 256],
        use_bn=True,
    ):
        """
        inp_dim : int - dimension d of the input
        hidden_dims1 : list - hidden layers (inluding d_out) of the first MLP block (defined with nn.Conv1d...)
        hidden_dims2 : list - hidden layers (without d_out) of the second MLP block
        use_bn       : bool - whether to use batchnorm
        """
        super().__init__()

        self.inp_dim = inp_dim
        self.use_bn = use_bn
        activation = nn.ReLU
        layers_list = []
        layers_list.append(
            torch.nn.Conv1d(
                in_channels=self.inp_dim, out_channels=hidden_dims1[0], kernel_size=1
            )
        )
        layers_list.append(activation())
        for k in range(len(hidden_dims1) - 1):
            if use_bn:
                layers_list.append(nn.BatchNorm1d(num_features=hidden_dims1[k]))

            layers_list.append(
                torch.nn.Conv1d(
                    in_channels=hidden_dims1[k],
                    out_channels=hidden_dims1[k + 1],
                    kernel_size=1,
                )
            )
            if k < len(layers_list) - 1:
                layers_list.append(activation())

        self.first_mlp = nn.Sequential(*layers_list)

        # a max pooling will happen

        # define second MLP
        layers_list = []
        layers_list.append(
            nn.Linear(in_features=hidden_dims1[-1], out_features=hidden_dims2[0])
        )
        layers_list.append(activation())
        for k in range(len(hidden_dims2) - 1):
            if use_bn:
                layers_list.append(nn.BatchNorm1d(num_features=hidden_dims2[k]))
            layers_list.append(
                nn.Linear(in_features=hidden_dims2[k], out_features=hidden_dims2[k + 1])
            )

            if k < len(layers_list) - 1:
                layers_list.append(activation())

        layers_list.append(
            nn.Linear(in_features=hidden_dims2[-1], out_features=inp_dim**2)
        )
        self.second_mlp = nn.Sequential(*layers_list)

    def forward(self, x):
        """
        x : (B, d, n) - This is standard shape for convolution inputs

        Output
        ------------
        (B, d , d) : output T defined as I + NET(x) for stability
        """
        X = self.first_mlp(x)
        # apply max pool
        X = torch.max(X, dim=-1).values
        # apply second MLP
        T = self.second_mlp(X)
        T = T.reshape((-1, self.inp_dim, self.inp_dim))
        return T + torch.eye(self.inp_dim, device=T.device).unsqueeze(0)

class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.first_transform = Tnet(
            inp_dim=3,
   
            use_bn=True,
        )  
        #* A 2 layers shared MLP over points to compute features of dim 64: ( 3,64) then 64,64
        self.first_mlp=MLP_con1D_Trick(
            inp_dim=3,
            hidden_dims1=[64,64],
            use_bn=True,
            activation=nn.ReLU
        )

        #* A 3 layers shared MLP over points to compute features of dim 1024 (64,128,1024)
        self.second_mlp=MLP_con1D_Trick(
            inp_dim=64,
            hidden_dims1=[64,64,128,1024],

            use_bn=True,
            activation=nn.ReLU
        )


        # finally a 3 layers MLP to compute the classification scores
        # YOUR CODE
        self.last_mlp=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,classes)
        )

    def forward(self, input):
        T = self.first_transform(input)  # transflation

        x = torch.einsum("ijk, ijl -> ijk", input, T)  ## TODO


        x=self.first_mlp(input)

        # T_feat = self.feature_transform(x)
        # x_feat = torch.einsum("ijk, ijl -> ijk", x, T_feat)  ## TODO



        x=self.second_mlp(x)
        # apply max_pooling as global feature vecotr

        # Decoder part 
        global_feature = torch.max(x, dim=-1).values
        out=self.last_mlp(global_feature)

        return out,T 

def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)
from tqdm import tqdm
from tqdm import trange

def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    t=trange(epochs)
    for epoch in t: 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            inputs=inputs.transpose(1,2)
            outputs = model(inputs)
            # outputs, m3x3 = model(inputs)
            loss = basic_loss(outputs, labels)
            # loss = pointnet_full_loss(outputs, labels, m3x3) #! PointNetfull
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs = model(inputs.transpose(1,2))
                    # outputs, __ = model(inputs.transpose(1,2)) #! For PointNetfull
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total

            # log it to be display with tqdm
            t.set_description(f'Epoch {epoch+1}')
            t.set_postfix(loss=loss.item(), test_acc=test_acc)
            # print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    if True: 
        # Basic Transformation
        train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())

    if False:
        # We added the scale transformation
        train_ds=PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms2())
    print(default_transforms2())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)
    if False:
        print("we are doing the model: MLP")
        model = MLP()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
        model.to(device)

        epochs=250
        train(model, device, train_loader, test_loader, epochs = epochs)

        t1 = time.time()
        print("Total time for training : ", t1-t0)

    if True:
        print("we are doing the model: PointNetBasic")
        model = PointNetBasic()
        print("model: PointNetBasic",model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
        model.to(device)
        
        epochs=250
        train(model, device, train_loader, test_loader, epochs = epochs)
        
        t1 = time.time()
        print("Total time for training : ", t1-t0)

    if True:
        print("we are doing the model: PointNetCls")
        model = PointNetFull()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
        model.to(device)
        
        epochs=250
        train(model, device, train_loader, test_loader, epochs = epochs)
        
        t1 = time.time()
        print("Total time for training : ", t1-t0)

    #model = PointNetBasic()
    #model = PointNetFull()
    

    
    


