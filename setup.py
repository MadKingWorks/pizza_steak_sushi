from ntpath import isdir
import  os
import argparse
from pathlib import Path
import requests,zipfile,io

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

    


