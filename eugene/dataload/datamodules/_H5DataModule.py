import torch
import os, pathlib, h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from ..dataloaders._SeqDataset import SeqDataset
from torchvision import transforms
from .._transforms import ToTensor

# From evoaug_analysis -- https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/utils.py
class H5DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, stage=None, lower_case=False, transpose=False, downsample=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.x = 'X'
        self.y = 'Y'
        if lower_case:
            self.x = 'x'
            self.y = 'y'
        self.transpose = transpose
        self.downsample = downsample
        self.transforms = transforms.Compose([ToTensor()])
        self.setup(stage)

    def setup(self, stage=None):
        # Assign train and val split(s) for use in DataLoaders
        if stage == "fit" or stage is None:
            with h5py.File(self.data_path, 'r') as dataset:
                x_train = np.array(dataset[self.x+"_train"]).astype(np.float32)
                x_valid = np.array(dataset[self.x+"_valid"]).astype(np.float32)
                if self.transpose:
                    x_train = np.transpose(x_train, (0,2,1))
                    x_valid = np.transpose(x_valid, (0,2,1))
                if self.downsample:
                    x_train = x_train[:self.downsample]
                    y_train = y_train[:self.downsample]
                self.x_train = x_train
                self.x_valid = x_valid
                self.y_train = np.array(dataset[self.y+"_train"]).astype(np.float32)
                self.y_valid = np.array(dataset[self.y+"_valid"]).astype(np.float32)
                self.num_classes = self.y_train.shape[1]
                if self.num_classes == 1:
                    self.y_train = np.squeeze(self.y_train)
                    self.y_valid = np.squeeze(self.y_valid)
            _, self.A, self.L = self.x_train.shape # N = number of seqs, A = alphabet size (number of nucl.), L = length of seqs

            
        # Assign test split(s) for use in DataLoaders
        if stage == "test" or stage is None:
            with h5py.File(self.data_path, "r") as dataset:
                x_test = np.array(dataset[self.x+"_test"]).astype(np.float32)
                if self.transpose:
                    x_test = np.transpose(x_test, (0,2,1))
                self.x_test = x_test
                self.y_test = np.array(dataset[self.y+"_test"]).astype(np.float32)
                self.num_classes = self.y_test.shape[1]
                if self.num_classes == 1:
                    self.y_test = np.squeeze(self.y_test)
            _, self.A, self.L = self.x_train.shape

            
    def train_dataloader(self):
        train_dataset = SeqDataset(self.x_train, targets=self.y_train, transform=self.transforms) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled
    
    def val_dataloader(self):
        valid_dataset = SeqDataset(self.x_valid, targets=self.y_valid, transform=self.transforms) 
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = SeqDataset(self.x_test, targets=self.y_test, transform=self.transforms)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 