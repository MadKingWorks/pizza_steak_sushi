"""
File: data_setup.py
 Author: sharma.uce@gmail.com
 Created: 2025-07-27
 Description:provides functions to download the data 
"""

from pathlib import Path

import requests
import os
from torch import device
import torch
import torchvision
import tqdm
from torchvision import transforms
from utils import create_data_loader
import torchinfo
import torch.nn as nn
def download_data(url:str,
                  filename:Path,
                  ):
    """Summary line.

    Args:
        url (str): Download URL
        filename:Qualified path of target file

    Returns:
        True: if Download was successfull , else False
    """

    try:
        request = requests.get(url)
        with open(filename,'wb') as f:
            #data = requests.get(request.content)
            f.write(request.content)
        return True
    except Exception as  e:
        print(f"The exception is {e}")
        return False

   

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from utils import uncompressfile
    cwd = Path(os.getcwd())
    datapath = cwd/"Data"
    filename = datapath/"pizza_sushi_steak.zip"
    if not os.path.exists(filename):
        print("This worked") if download_data(url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",filename=filename) else print("This did not work")
    else:
        print(f"The file already exists ")
    try:
        uncompressfile(filename)
        print(f"The uncompression worked !")
    except Exception as e:
        print(f"The exception returned is {e}")

    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    manual_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    cwd = Path(os.getcwd())

    auto_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
    train_data_path = cwd/"Data//train"
    test_data_path = cwd/"Data//test"

    train_data_loader,test_data_loader , classes = create_data_loader(train_dir=train_data_path,test_dir=test_data_path,transform=auto_transform,batch_size=32,num_workers=0,)

    print(f"The manual transform is {manual_transform}")
    print(f"The classes are {classes}")
    print(f"The auto transform is {auto_transform}")
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights).to(device)
    torchinfo.summary(model=model,
                      input_size=(32,3,224,224),
                      col_names=["input_size","output_size","num_params","trainable"],
                      col_width=20,
                      row_settings=["var_names"])
    
    for param in model.features.parameters():
        param.requires_grad = False

    output_Shape = len(classes)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2,inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_Shape,
                        bias=True)
    ).to(device)
    
    torchinfo.summary(model=model,
                      input_size=(32,3,224,224),
                      col_names=["input_size","output_size","num_params","trainable"],
                      col_width=20,
                      row_settings=["var_names"]
                      )
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    from engine import train
    resuls = train(model,
                   train_data_loader,
                   test_data_loader,
                   optimizer=optimizer,
                   loss_fn=loss_fn)
    
    

    
    
    

    

    

    

    
