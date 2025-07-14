import  os
#create the structure of the project
#setup the data folder
from pathlib import Path

path  = os.getcwd()
data_path = path / Path("/Data")
if os.path.isdir(data_path) :
    print(f"The folder {data_path} exists and will not be created again")
else:
    print(f"The folder {data_path} will be created ...")
    os.mkdir(data_path)



