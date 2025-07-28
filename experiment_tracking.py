"""
File: experiment_tracking.py
 Author: sharma.uce@gmail.com
 Created: 2025-07-28
 Description: Description
"""
import os
from pathlib import Path
from random import weibullvariate
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import torchvision
from torch import  nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from engine import train
from utils import set_device, set_seeds
from data_setup import create_data_loader
from torch.utils.tensorboard import SummaryWriter


device= set_device()
set_seeds()


cwd = os.getcwd()
print(f"Curren directory is {cwd}")
train_data_path = Path(cwd)/"Data//train"
test_data_path = Path(cwd)/"Data//test"





auto_model_b0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT    #get best weights
auto_model_b0_transform = auto_model_b0_weights.transforms()

auto_model_b1_weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT    #get best weights
auto_model_b1_transform = auto_model_b1_weights.transforms()

auto_model_b2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT    #get best weights
auto_model_b2_transform = auto_model_b2_weights.transforms()

auto_model_b3_weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT    #get best weights
auto_model_b3_transform = auto_model_b3_weights.transforms()

train_dataloader_b0,test_dataloader_b0  , data_classes_b0  = create_data_loader(
    train_dir=train_data_path,
    test_dir=test_data_path,
    transform=auto_model_b0_transform,
    batch_size=32,
    num_workers=0,
)


train_dataloader_b1,test_dataloader_b1  , data_classes_b1  = create_data_loader(
    train_dir=train_data_path,
    test_dir=test_data_path,
    transform=auto_model_b1_transform,
    batch_size=32,
    num_workers=0,
)

train_dataloader_b2,test_dataloader_b2  , data_classes_b2  = create_data_loader(
    train_dir=train_data_path,
    test_dir=test_data_path,
    transform=auto_model_b2_transform,
    batch_size=32,
    num_workers=0,
)

train_dataloader_b3,test_dataloader_b3  , data_classes_b3  = create_data_loader(
    train_dir=train_data_path,
    test_dir=test_data_path,
    transform=auto_model_b3_transform,
    batch_size=32,
    num_workers=0,
)

#provide sumamry of all the models


auto_model_b0 = torchvision.models.efficientnet_b0(weights=auto_model_b0_weights)
auto_model_b1 = torchvision.models.efficientnet_b1(weights=auto_model_b1_weights)
auto_model_b2 = torchvision.models.efficientnet_b2(weights=auto_model_b2_weights)
auto_model_b3 = torchvision.models.efficientnet_b3(weights=auto_model_b3_weights)


print(f"<<<<<<<auto_model_b0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
summary(model=auto_model_b0,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )


print(f"<<<<<<<<<<<<auto_model_b1<<<<<<<<<<<<<<<<<<<<<<<<")
summary(model=auto_model_b1,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )


print(f"<<<<<<<<<<<<auto_model_b2<<<<<<<<<<<<<<<<<<<<<<<<")

summary(model=auto_model_b2,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )

print(f"<<<<<<<<<<<<auto_model_b3<<<<<<<<<<<<<<<<<<<<<<<<")

summary(model=auto_model_b3,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )



#freeze the features of all models

for param in auto_model_b0.features.parameters():
    param.requires_grad = False
for param in auto_model_b1.features.parameters():
    param.requires_grad = False

for param in auto_model_b2.features.parameters():
    param.requires_grad = False

for param in auto_model_b3.features.parameters():
    param.requires_grad = False


# create a custom classifier as per our data

auto_model_b0.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=len(data_classes_b0),
                    bias=True)).to(device)

auto_model_b1.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=len(data_classes_b1),
                    bias=True)).to(device)

auto_model_b2.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1408,
                    out_features=len(data_classes_b2),
                    bias=True)).to(device)

auto_model_b3.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2,inplace=True),
    torch.nn.Linear(in_features=1536,
                    out_features=len(data_classes_b3),
                    bias=True)).to(device)



    

print(f"<<<<<<<auto_model_b0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
summary(model=auto_model_b0,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )


print(f"<<<<<<<<<<<<auto_model_b1<<<<<<<<<<<<<<<<<<<<<<<<")
summary(model=auto_model_b1,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )


print(f"<<<<<<<<<<<<auto_model_b2<<<<<<<<<<<<<<<<<<<<<<<<")

summary(model=auto_model_b2,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )

print(f"<<<<<<<<<<<<auto_model_b3<<<<<<<<<<<<<<<<<<<<<<<<")

summary(model=auto_model_b3,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable",],
        col_width=20,
        row_settings = ["var_names"],
        )

loss_fn = nn.CrossEntropyLoss()

auto_model_b0_optimizer = torch.optim.Adam(auto_model_b0.parameters(),lr=0.001)
auto_model_b1_optimizer = torch.optim.Adam(auto_model_b1.parameters(),lr=0.001)
auto_model_b2_optimizer = torch.optim.Adam(auto_model_b2.parameters(),lr=0.001)
auto_model_b3_optimizer = torch.optim.Adam(auto_model_b3.parameters(),lr=0.001)

writer = SummaryWriter()

auto_model_b0_training_results = train(auto_model_b0,
                                       train_dataloader_b0,
                                       test_dataloader_b0,
                                       auto_model_b0_optimizer,
                                       writer=writer,
                                       loss_fn=loss_fn,
                                       epochs=10,
                                       )

print(auto_model_b0_training_results)
auto_model_b1_training_results = train(auto_model_b1,
                                       train_dataloader_b1,
                                       test_dataloader_b1,
                                       auto_model_b1_optimizer,
                                       writer=writer,
                                       loss_fn=loss_fn,
                                       epochs=10,
                                       )

print(auto_model_b1_training_results)
auto_model_b2_training_results = train(auto_model_b2,
                                       train_dataloader_b2,
                                       test_dataloader_b2,
                                       auto_model_b2_optimizer,
                                       writer=writer,
                                       loss_fn=loss_fn,
                                       epochs=10,
                                       )

print(auto_model_b2_training_results)
auto_model_b3_training_results = train(auto_model_b3,
                                       train_dataloader_b3,
                                       test_dataloader_b3,
                                       auto_model_b3_optimizer,
                                       writer=writer,
                                       loss_fn=loss_fn,
                                       epochs=10,
                                       )

print(auto_model_b3_training_results)
