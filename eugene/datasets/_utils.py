from pathlib import Path
from functools import wraps
import os, gzip, wget, io
import pandas as pd
from .._settings import settings
HERE = Path(__file__).parent


def check_datasetdir_exists(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        settings.datasetdir.mkdir(exist_ok=True)
        return f(*args, **kwargs)

    return wrapper


def deBoerCleanup(file: pd.DataFrame, index: int) -> pd.DataFrame:
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


def try_download_urls(data_idxs: list, url_list: list, ds_name: str, compression: str = "") -> list:
    ds_path = os.path.join(HERE.parent, settings.datasetdir, ds_name)
    paths = []
    if compression != "":
        compression = "." + compression
    for i in data_idxs:
        base_name = os.path.basename(url_list[i]).split(".")[0] + f".csv{compression}"
        search_path = os.path.join(HERE.parent, settings.datasetdir, ds_name, base_name)
        if not os.path.exists(search_path):
            if not os.path.isdir(ds_path):
                print(f"Path {ds_path} does not exist, creating new folder.")
                os.makedirs(ds_path)

            print(f"Downloading {ds_name} {os.path.basename(url_list[i])} to {ds_path}...")
            print(url_list[i], os.path.relpath(ds_path))
            path = wget.download(url_list[i], os.path.relpath(ds_path))
            paths.append(path)
            print(f"Finished downloading {os.path.basename(url_list[i])}")

            if compression == ".gz":
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
                os.remove(os.path.join(ds_path, os.path.basename(url_list[i])))
            else:
                # Implement when needed
                pass
        else:
            print(f"Dataset {ds_name} {base_name} has already been dowloaded.")
            paths.append(os.path.join(ds_path, base_name))

    return paths
