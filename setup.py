from ntpath import isdir
import  os
import argparse
from pathlib import Path
import requests,zipfile,io
import torch
from torchinfo import summary
from torch import nn
from engine import train
from torch.utils.data import DataLoader, dataset
from utils import plot_transformed_images, walk_through_dir ,plot_random_image ,create_data_loader
from torchvision import datasets, transforms

from model_builder import TinyVGG
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument('cwd')
parser.add_argument('image_size_transformed')
parser.add_argument('BATCH_SIZE')
parser.add_argument('lr')
parser.add_argument('NUM_EPOCHS')

args=parser.parse_args()
#create the structure of the project
#setup the data folder

print(f"parsed address is {args.cwd}")
path  = Path(args.cwd)
image_size_transformed = int(args.image_size_transformed)
BATCH_SIZE = int(args.BATCH_SIZE)
lr = float(args.lr)
data_path = path / Path("Data")

NUM_EPOCHS = int(args.NUM_EPOCHS) 
print(f"parsed address is {data_path}")
if os.path.isdir(data_path) :
    print(f"The folder {data_path} exists and will not be created again")
else:
    print(f"The folder {data_path} will be created ...")
    os.mkdir(data_path)

#download the data from mrdbourke site
mrdbourkedata = r"https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"
#the file will be donaloaded in the location Data and then unzipped

#check if the Data folder exists

if os.path.isdir(data_path):
    if not os.listdir(data_path):
        request = requests.get(mrdbourkedata)
        zipfiles = zipfile.ZipFile(io.BytesIO(request.content))
        zipfiles.extractall(data_path)
    else:
        print(f"The data files are present in the folder {data_path} so download will not take place")

train_data_path = data_path/"train"
test_data_path = data_path/"test"

    
#walk_through_dir(data_path)
#plot_random_image(data_path,43,5)

data_transform = transforms.Compose([
    transforms.Resize(size =(image_size_transformed,image_size_transformed)),    # Resize any input image
    transforms.RandomHorizontalFlip(p=0.5),    #randomly flip horizontally to get data augmentation
    transforms.ToTensor()    #converts the image to a tensor so as for training 
])

list_of_train_data_path = list(train_data_path.glob("**/*/*.jpg"))
#print(f"path  of train_data_path is {train_data_path} and contains {list_of_train_data_path} elements ")
#print(f"Type of list_of_train_data_path is {type(list_of_train_data_path)} and contains {len(list_of_train_data_path)} elements ")
#plot_transformed_images(list_of_train_data_path,
#                        data_transform,
#                        3,
#                        42)

list_of_test_data_path = list(test_data_path.glob("**/*/*.jpg"))
#print(f"path  of test_data_path is {test_data_path} and contains {list_of_test_data_path} elements ")
#plot_transformed_images(list_of_test_data_path,
#                        data_transform,
#                        3,
#                        42)



#create data sets

train_dataset = datasets.ImageFolder(root=train_data_path,
                                  transform=data_transform,
                                  target_transform=None
                                  )

test_dataset = datasets.ImageFolder(root=test_data_path,
                                    transform=data_transform,
                                    target_transform=None)

train_data_dataloader = DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   num_workers=0,
                                   shuffle=True
                                   
                                   )

test_dataloader = DataLoader(dataset = test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=0,
                             shuffle=False)

#for img , label in test_dataloader:
#    img_simple,label_simple  = img[0].unsqueeze(dim=0),label[0]
#    print(f"Image label : {label} Image shape : {img.shape} -> [batch_size , color_channels,height , width]")
#    print(f"Label shape {label.shape}")
    

model_0 = TinyVGG(3,
                  10,
                  len(train_dataset.classes),
                  3,
                  2,
                  1)

#print(model_0)
#print(f"model output logits are {model_0(img_simple)} ")
#print(f"prediction probabilities are {torch.softmax(model_0(img_simple),dim=1)}")
#print(f"prediction output label is {torch.argmax(torch.softmax(model_0(img_simple),dim=1))}")
#print(f"Original label is {label_simple}")

#print(summary(model_0,input_size = [1,3,image_size_transformed,image_size_transformed]).encode("utf-8"))
#summary_file = "summary.log"
#with open(summary_file,'ear',encoding='utf-8') as f:
#    f.write(summary(model_0,input_size=[1,3,image_size_transformed,image_size_transformed#]))

torch.manual_seed(42)
torch.cuda.manual_seed(42)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params = model_0.parameters(),
                             lr = lr)

start_time = timer()
model_0_results = train(model=model_0,
                        train_dataloader=train_data_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs = NUM_EPOCHS)


end_time = timer()

print(f"TOTAL TRAINING TIME : {end_time - start_time:.3f} seconds")

