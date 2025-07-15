import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
from PIL import Image
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader



def plot_random_image(data_path : Path,random_seed:int,num:int):
    
    """Connects to the next available port.

    Args:
      data_path: path of the folder where images are located
      random_seed : provide an int to replicate randomness
      num : number if random images to be checked. this should not be greater than total images in the folder, default :10

    Returns:
      None

    Raises:
      None
    """

    #image_list =
    random.seed(random_seed)
    #get a list of all images
    list_of_image_paths = list(data_path.glob("*/*/*.jpg"))
    #select random number of images
    random_path_of_images = random.choice(list_of_image_paths)
    #get image class from directory
    image_class = random_path_of_images.parent.stem
    img = Image.open(random_path_of_images)
    print(f"Random image path : {random_path_of_images}")
    print(f"Image class : {image_class}")
    print(f"Image height is :{img.height}")
    print(f"Image width is :{img.width}")
    img.show()
    
    
    

def walk_through_dir(path):
    for (root,dir,file) in os.walk(path):
        print(f"We have {len(dir)} directories and {len(file)} files at location {root}")


def create_data_loader(train_dir:Path,
                       test_dir:Path,
                       transform:transforms.Compose,
                       batch_size:int,
                       num_workers:int,
                       ):
    """
    #TODO

    """
    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)
    class_names = train_data.classes
    class_to_idx = train_data.class_to_idx
    #TURN IMAGES TO DATALOADERS

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader,test_dataloader,class_names

   
    

    
    
