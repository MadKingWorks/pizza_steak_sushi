"""
File: vit.py
 Author: sharma.uce@gmail.com
 Created: 2025-08-02
 Description: Description
"""


"""
1. get the data
2. Create  the dataloaders
3. train the model
4. Get the plots for perfomance of the model

"""
from pathlib import Path
from data_setup import create_data_loader
from torchvision import transforms
from utils import  list_of_path, plot_random_image, plot_transformed_images
data_path = Path("/home/remote/pizza_steak_sushi/Data/train")
test_path = Path("/home/remote/pizza_steak_sushi/Data/test")
IMG_SIZE = 224    #table 3 of the paper
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

print(f"Manually created transformer is {manual_transform}")



data_loader = create_data_loader(
    train_dir=data_path,
    test_dir=test_path,
    transform=manual_transform,
    batch_size=32,
    num_workers=0,
)

#plot_random_image(data_path=data_path,


#                  random_seed=42,
#                  num=1)


images  = list_of_path(data_path)
plot_transformed_images(images,
                        manual_transform,
                        1,
                        43,
                        )


PATCH_SIZE=16

number_of_patches = IMG_SIZE * IMG_SIZE / PATCH_SIZE**2


print(f"The number of pathches = {number_of_patches}")


