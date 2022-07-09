from pathlib import Path
from functools import wraps
from .._settings import settings

import os, gzip, wget, io

import pandas as pd

HERE = Path(__file__).parent
def check_datasetdir_exists(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        settings.datasetdir.mkdir(exist_ok=True)
        return f(*args, **kwargs)

    return wrapper

def deBoerCleanup(file: pd.DataFrame, index: int) -> pd.DataFrame:
    if index == 5:
        # 
        return file
    elif index == 6:
        # Remove upper title row, keep only 1 of 4 data columns
        return file
    elif index == 7:
        # Remove upper title row, data potentially means something different because its the only single data csv with a title?
        return file
    else:
        return file

def try_download_urls(data_idxs: list, url_list: list, ds_name: str, is_gz: bool = False) -> list:
    paths = []
    for i in data_idxs:
        csv_name = os.path.basename(url_list[i]).split(".")[0] + ".csv"
        if not os.path.exists(os.path.join(HERE.parent, settings.datasetdir, ds_name, csv_name)):
            ds_path = os.path.join(HERE.parent, settings.datasetdir, ds_name)
            if not os.path.isdir(ds_path):
                print(f"Path {ds_path} does not exist, creating new folder.")
                os.mkdir(ds_path)

            print(f"Downloading {ds_name} {os.path.basename(url_list[i])} to {ds_path}...")
            path = wget.download(url_list[i], os.path.relpath(ds_path))
            print(f"Finished downloading {os.path.basename(url_list[i])}")

            if is_gz:
                print("Processing gzip file...")
                with gzip.open(path) as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8") as file:
                        file = pd.read_csv(file, delimiter=r"\t", engine="python")

                        if ds_name == "deBoer20":
                            file = deBoerCleanup(file, i)

                        save_path = os.path.join(ds_path, csv_name)
                        print(f"Saving csv file to {save_path}...")
                        file.to_csv(save_path, index = False)
                        print(f"Saved csv file to {save_path}")
                        paths.append(save_path)
                os.remove(os.path.join(ds_path, os.path.basename(url_list[i])))
            else:
                # If file is not packed, same logic but without gzip, implement when needed
                pass
        else:
            print(f"Dataset {ds_name} {csv_name} has already been dowloaded.")
            paths.append(os.path.join(HERE.parent, settings.datasetdir, ds_name, csv_name))
    return paths
