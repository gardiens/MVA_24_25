# import numpy as np

import torch

from torchvision import datasets, transforms, models
from torch.functional import F
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
def precompute_features(
    model: models.ResNet, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is 
    the rest of the model, it is not necessary to recompute $g(x)$ at 
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and 
    create a new dataset 
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation
    
    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Remove the last fully connected layer to extract features
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    features = []
    labels = []

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():  # No need to track gradients
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten to vector
            features.append(outputs.cpu())
            labels.append(targets)

    # Concatenate all collected features and labels
    features = torch.cat(features)
    labels = torch.cat(labels)
    return TensorDataset(features, labels)


class LastLayer(nn.Module):
    def __init__(self,num_classes:int=2):

        super(LastLayer, self).__init__()
        self.num_classes = num_classes
        self.fc=nn.Linear(512, num_classes,bias=True) #can be 2048 if using resnet50
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        return self.fc(x)
        raise NotImplementedError("Implement the forward pass of the LastLayer module")

from torchvision.models import resnet18

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        self.base_model=resnet18(weights="DEFAULT")
        self.base_model.fc=LastLayer()
        # <YOUR CODE>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <YOUR CODE>
        x = self.base_model(x)
        return x