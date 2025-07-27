"""
File: data_setup.py
 Author: sharma.uce@gmail.com
 Created: 2025-07-27
 Description:provides functions to download the data 
"""

from pathlib import Path

import requests
import os
import tqdm


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
    
