from typing import Dict
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
from PIL import Image
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from typing import Dict
import zipfile

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

def plot_transformed_images(image_paths : list[Path] ,
                            transform : transforms.Compose ,
                            n=3,
                            seed=42):
    """Function.

    Args:
        image_paths : List of paths from where images will be selected randomly
    
        transform : transformation to be applied before showing the image
    
        n : number of images to be plotted
    
        seed : random seed to reproduce the results

    Returns:
        Description : None 
    """

    random.seed(seed)
    random_image_paths = random.sample(image_paths ,k=n )
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig , ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \n Size {f.size}")
            ax[0].axis("off")

            #apply transformation and then plot the image
            transformed_image = transform(f).permute(1,2,0)    # matplotlib uses color code first
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed Image \n Size : {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f" Image Class : {image_path.parent.stem}",fontsize=16)
            plt.show()


def plot_loss_curves(results ):
    """Function to plot the loss curves as per inputs

    Args:
         results :Dict(str,list[float]) a result dictionalry with various data

    Returns:
        Description.
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(loss))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label="train_loss")
    plt.plot(epochs,test_loss,label="test_loss")
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy,label="train_acc")
    plt.plot(epochs,test_accuracy,label="test_acc")
    plt.title('accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
    
       
def uncompressfile(filename):
    """uncompress file at the same location.

    Args:
        filename(str):filename.

    Returns:
        True: if thr file got uncompressed successfully else False
    """
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(filename.parent)
    
        
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
def set_seeds(seed = 42):
    """sets the seeds ; default value is 42

    Args:
        seed (int): random seed int value.

    Returns:
        None.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

    
    
    
