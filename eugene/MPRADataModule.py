import pytorch_lightning as pl
from MPRADataset import MPRADataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from transforms import ReverseComplement, Augment, OneHotEncode, ToTensor
from load_data import load

class MPRADataModule(pl.LightningDataModule):
    def __init__(self, seq_file: str, batch_size: int = 32, num_workers: int = 0, transform=None, split=0.9, load_kwargs={}):
        super().__init__()
        self.seq_file = seq_file
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([Augment(randomize_linker_p=0.1, enhancer="WT-otx-a"),
                                                 ReverseComplement(ohe_encoded=False), 
                                                 OneHotEncode(), 
                                                 ToTensor(transpose=True)])
        else:
            self.transform = transform
        self.load_kwargs = load_kwargs
        self.num_workers = num_workers
        
    def setup(self, stage: str = None) -> None:
        seqs, targets = load(self.seq_file, **self.load_kwargs)
        dataset = MPRADataset(seqs, targets, transform=self.transform)
        dataset_len = len(dataset)
        train_len = int(dataset_len*0.9)
        val_len = dataset_len - train_len
        self.train, self.val = random_split(dataset, [train_len, val_len])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers
        )
