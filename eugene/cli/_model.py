import importlib
from ..dataloading.dataloaders import SeqDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from ..models import FCN, CNN, RNN, Hybrid


if __name__ == "__main__":
    model_type = "hybrid"
    module = getattr(importlib.import_module("eugene.models"), model_type)
    cli = LightningCLI(module, SeqDataModule, save_config_overwrite=True)
