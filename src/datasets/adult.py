import os
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from src.datasets.dataset_utils import SubpopDataset

import torch
import numpy as np

class Adult(SubpopDataset):
    def __init__(self,
                 train=True,
                 target_attr_name="Income Binary",
                 group_attr_name=["sex"],
                 seed=0
                 ):
        download_adult_dataset()
        dataset_orig = load_preproc_data_adult()
        df, d = dataset_orig.convert_to_dataframe()
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)
        if train:
            df = train_df
        else:
            df = test_df
        
        self.y_list = df[target_attr_name].to_numpy().astype(int)
        if len(group_attr_name)==1:
            group_attr_name = group_attr_name[0]
            self.g_list = df[group_attr_name].to_numpy().astype(int)
        else:
            raise NotImplementedError("combining attributes not implemented yet")
        drop_colums = set([target_attr_name]).union(set(["race","sex"]))
        df_temp = df.drop(columns=drop_colums)
        self.x_array = df_temp.to_numpy().astype(np.float32)
        super().__init__()

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, index):
        x_features, target, attr = self.x_array[index], self.y_list[index], self.g_list[index]
        return index, torch.tensor(x_features) , (torch.tensor(target), torch.tensor(attr))

def download_adult_dataset():
    """
    Checks if the Adult dataset files are present in the aif360 data directory.
    If not, it downloads them from the UCI Machine Learning Repository.
    """
    # Determine the target directory where aif360 expects the data.
    # We prioritize finding the aif360 package's actual installation path.
    # If aif360 is not found, we fall back to the path provided in the error message.
    target_dir = None
    try:
        import aif360
        # Get the path to the aif360 package
        aif360_package_path = os.path.dirname(os.path.abspath(aif360.__file__))
        target_dir = os.path.join(aif360_package_path, 'data', 'raw', 'adult')
        print(f"Detected aif360 package path: {aif360_package_path}")
    except ImportError:
        print("aif360 not found. Please ensure aif360 is installed.")
        print("Attempting to use the path from the error message as a fallback.")
        # Fallback if aif360 is not installed or import fails
        # This path is derived directly from the user's error message.
        target_dir = '/usr/local/lib/python3.11/dist-packages/aif360/data/raw/adult'

    if target_dir is None:
        print("Could not determine the target directory for the dataset. Exiting.")
        return

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        print(f"Creating directory: {target_dir}")
        os.makedirs(target_dir)
    else:
        print(f"Directory already exists: {target_dir}")

    # List of files to download and their respective URLs
    files_to_download = {
        'adult.data': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'adult.test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        'adult.names': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
    }

    # Iterate through the files and download if missing
    for filename, url in files_to_download.items():
        filepath = os.path.join(target_dir, filename)

        if os.path.exists(filepath):
            print(f"'{filename}' already exists at '{filepath}'. Skipping download.")
        else:
            print(f"Downloading '{filename}' from '{url}' to '{filepath}'...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise an exception for HTTP errors

                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()

                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong during download.")
                else:
                    print(f"Successfully downloaded '{filename}'.")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading '{filename}': {e}")
            except IOError as e:
                print(f"Error writing file '{filename}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    print("\nAdult dataset download check complete.")

