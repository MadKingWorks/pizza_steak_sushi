from ntpath import isdir
import  os
import argparse
from pathlib import Path
import requests,zipfile,io
from utils import walk_through_dir ,plot_random_image ,create_data_loader
from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument('cwd')
args=parser.parse_args()
#create the structure of the project
#setup the data folder

print(f"parsed address is {args.cwd}")
path  = Path(args.cwd)
data_path = path / Path("Data")

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

    
walk_through_dir(data_path)
plot_random_image(data_path,43,5)
transform = transforms.Compose(
    
)

train_data_loader , test_data_loader , classes = create_data_loader(
    train_data_path,
    test_data_path,
    
)
