
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import Subset
from PIL import Image
import json
import time

normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])

# Build and train your network
def create_network(model_name='resnet50', output_size=102, hidden_layers=[1000]):
    if model_name == 'resnet50':
        # Download the model
        model = models.resnet50(pretrained=True)
        # Replace the model classifier
        model.fc = create_classifier(2048, output_size, hidden_layers)
        
        return model
        
    if model_name == 'resnet152':
        # Download the model
        model = models.resnet152(pretrained=True)
        # Replace the model classifier
        model.fc = create_classifier(2048, output_size, hidden_layers)
        
        return model
        
    return None

    # Create a new classifier
def create_classifier(input_size, output_size, hidden_layers=[], dropout=0.5,
                      activation=nn.RReLU(), output_function=nn.LogSoftmax(dim=1)):
    dict = OrderedDict()
    
    if len(hidden_layers) == 0:
        dict['layer0'] = nn.Linear(input_size, output_size)

    else:
        
        dict['layer0'] = nn.Linear(input_size, hidden_layers[0])
        if activation:
            dict['activ0'] = activation
        if dropout:
            dict['drop_0'] = nn.Dropout(dropout)
        
        #for layer_in, layer_out in range(len(hidden_layers)):
        for layer, layer_in in enumerate(zip(hidden_layers[:-1],hidden_layers[1:])):
            dict['layer'+str(layer+1)] = nn.Linear(layer_in[0],layer_in[1])
            if activation:
                dict['activ'+str(layer+1)] = activation
            if dropout:
                dict['drop_'+str(layer+1)] = nn.Dropout(dropout)
            
        dict['output'] = nn.Linear(hidden_layers[-1], output_size)

    if output_function:
        dict['output_function'] = output_function
    
    return nn.Sequential(dict)

    # Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path='checkpoint.pt'):
    checkpoint = torch.load(checkpoint_path)

    model_name = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    
    model = create_network(model_name=model_name,
                         output_size=output_size, hidden_layers=hidden_layers)
   
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_label_to_name = checkpoint['cat_label_to_name']
    
    return model

model = load_checkpoint('C:/Users/96399/Desktop/Django_project/graduation/models/checkpoint.pt')