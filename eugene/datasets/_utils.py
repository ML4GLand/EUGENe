from pathlib import Path
from functools import wraps
import os, gzip, wget, io
import pandas as pd
import numpy as np
import torch
from typing import Union
from .._settings import settings
HERE = Path(__file__).parent


def check_dataset_dir_exists(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        settings.dataset_dir.mkdir(exist_ok=True)
        return f(*args, **kwargs)
    return wrapper


def deBoerCleanup(
    file: pd.DataFrame, 
    index: int
) -> pd.DataFrame:
    """Cleanup the deBoer dataset to remove the first column and the last row.

    Parameters
    ----------
    file : pd.DataFrame
        The dataset to be cleaned.
    index : int
        The index of the dataset to be cleaned.
    """
    if index == 5:
        # Remove upper title row, keep only expression column
        file = file.drop(index=0, columns=[0,3,4])
        return file
    elif index == 6:
        # Remove upper title row, keep only 1 of 4 data columns
        file = file.drop(index=0, columns=[1,2,3,4])
        return file
    elif index == 7:
        # Remove upper title row
        file = file.drop(index=0)
        return file
    else:
        return file

def random_ohe_seqs (
    seq_len: int,
    batch_size: int = 1,
    return_tensor: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
) -> Union[np.ndarray, torch.tensor]:

    rand = np.rot90(np.rot90(np.eye(4)[np.random.choice(4, (seq_len, batch_size))]), axes=[1,2])

    if return_tensor:
        rand = torch.from_numpy(rand.copy()).to(device).type(dtype)

    return rand


def try_download_urls(
    data_idxs: list, 
    url_list: list, 
    ds_name: str, 
    processing: str = None
) -> list:
    """Download the data from the given urls.

    Parameters
    ----------
    data_idxs : list
        The indices of the data to be downloaded.
    url_list : list
        The urls of the data to be downloaded.
    ds_name : str
        The name of the dataset to be downloaded.
    processing : str, optional
        The processing of the data to be downloaded. The default is None.
    
    Returns
    -------
    list
        The downloaded data.
    """
    ds_path = os.path.join(HERE.parent, settings.dataset_dir, ds_name)
    paths = []
    if processing is not None:
        processing = "." + processing
    for i in data_idxs:
        base_name = os.path.basename(url_list[i]).split("?")[0] #.split(".")[0] + f".csv{processing}"
        search_path = os.path.join(HERE.parent, settings.dataset_dir, ds_name, base_name)
        if not os.path.exists(search_path):
            if not os.path.isdir(ds_path):
                print(f"Path {ds_path} does not exist, creating new folder.")
                os.makedirs(ds_path)

            print(f"Downloading {ds_name} {os.path.basename(url_list[i])} to {ds_path}...")
            path = wget.download(url_list[i], os.path.relpath(ds_path))
            print(f"Finished downloading {os.path.basename(url_list[i])}")

            # Remove this sometime in the future for cleanliness? Only used for deBoer and could be worked around, would have to do cleanup after loading instead.
            if processing == ".gz":
                print("Processing gzip file...")
                with gzip.open(path) as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8") as file:
                        file = pd.read_csv(file, delimiter=r"\t", engine="python", header=None)

                        if ds_name == "deBoer20":
                            file = deBoerCleanup(file, i)

                        save_path = os.path.join(ds_path, base_name)
                        print(f"Saving file to {save_path}...")
                        file.to_csv(save_path, index = False, compression="gzip")
                        print(f"Saved file to {save_path}")
                        paths.append(save_path)
                #os.remove(os.path.join(ds_path, os.path.basename(url_list[i])))
            else:
                paths.append(path)
        else:
            print(f"Dataset {ds_name} {base_name} has already been downloaded.")
            paths.append(os.path.join(ds_path, base_name))

    return paths
