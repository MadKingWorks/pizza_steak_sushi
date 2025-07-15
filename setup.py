import  os
import argparse
from pathlib import Path

args = argparse.ArgumentParser()

cwd = args.add_argument()
#create the structure of the project
#setup the data folder


path  = Path()
data_path = path / Path("/Data")
if os.path.isdir(data_path) :
    print(f"The folder {data_path} exists and will not be created again")
else:
    print(f"The folder {data_path} will be created ...")
    os.mkdir(data_path)



