import numpy as np
from os import PathLike
from typing import Union, List
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule, PopulationBasedTraining
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch 
from ray.tune.search.hyperopt import HyperOptSearch
from ..dataload import SeqData, SeqDataset
from .._settings import settings
from ..models import get_model, init_weights


def hyperopt_with_tune(
    config: dict,
    sdata: SeqData = None,
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    epochs: int = 10,
    gpus: int = None,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataset: SeqDataset = None,
    val_dataset: SeqDataset = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    transform_kwargs: dict = {},
    seed: int = None,
    verbosity = None,
):
    model = get_model(config["arch"], config)
    init_weights(model)
    gpus = gpus if gpus is not None else settings.gpus
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    log_dir = log_dir if log_dir is not None else settings.logging_dir
    name = name if name is not None else config["arch"]
    seed_everything(seed, workers=True) if seed is not None else seed_everything(settings.seed)
    batch_size = config["batch_size"]
    if train_dataloader is not None:
        assert val_dataloader is not None
    elif train_dataset is not None:
        assert val_dataset is not None
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
    elif sdata is not None:
        assert target_keys is not None
        targs = sdata.seqs_annot[target_keys].values  
        if len(targs.shape) == 1:
            nan_mask = np.isnan(targs)
        else:
            nan_mask = np.any(np.isnan(targs), axis=1)
        print(f"Dropping {nan_mask.sum()} sequences with NaN targets.")
        sdata = sdata[~nan_mask]
        train_idx = np.where(sdata.seqs_annot[train_key] == True)[0]
        train_dataset = sdata[train_idx].to_dataset(
            target_keys=target_keys,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        train_dataloader = train_dataset.to_dataloader(
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=True
        )
        val_idx = np.where(sdata.seqs_annot[train_key] == False)[0]
        val_dataset = sdata[val_idx].to_dataset(
            target_keys=target_keys,
            seq_transforms=seq_transforms,
            transform_kwargs=transform_kwargs,
        )
        val_dataloader = val_dataset.to_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        raise ValueError("No data provided to train on.")
    logger = TensorBoardLogger(log_dir, name=name, version=version)
    callbacks = []
    metrics = {"loss": "val_loss"}
    callbacks.append(TuneReportCallback(metrics, on="validation_end"))
    trainer = Trainer(
        max_epochs=epochs,
        gpus=gpus,
        logger=logger,
        progress_bar_refresh_rate=0,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

scheduler_dict = {
    "ASHAScheduler": ASHAScheduler,
    "MedianStoppingRule": MedianStoppingRule,
    "PopulationBasedTraining": PopulationBasedTraining,
    
}
default_scheduler_args = {
    "ASHAScheduler": {"metric":"loss", "mode":"min", "max_t":10, "grace_period":1, "reduction_factor":2},
    "MedianStoppingRule": {"metric":"loss", "mode":"min", "grace_period":1},
    "PopulationBasedTraining": {"metric":"loss", "mode":"min", "grace_period":1},
}
algo_dict = {
    "BayesOptSearch": BayesOptSearch,
    "HyperOptSearch": HyperOptSearch,
    "BasicVariantGenerator": BasicVariantGenerator,
}
default_algo_args = {
    "BayesOptSearch": {"metric":"loss", "mode":"min"},
    "HyperOptSearch": {"metric":"loss", "mode":"min"},
    "BasicVariantGenerator": {},
}

def hyperopt(
    config,
    sdata = None,
    target_keys: Union[str, List[str]] = None,
    train_key: str = "train_val",
    scheduler="ASHAScheduler",
    algorithm="BasicVariantGenerator",
    num_samples: int = 10,
    epochs: int = 10,
    gpus: int = None,
    cpus: int = 1,
    num_workers: int = None,
    log_dir: PathLike = None,
    name: str = None,
    version: str = None,
    train_dataset: SeqDataset = None,
    val_dataset: SeqDataset = None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    seq_transforms: List[str] = None,
    transform_kwargs: dict = {},
    seed: int = None,
    verbosity = None,
    scheduler_kwargs: dict = None,
    algorithm_kwargs: dict = None,
    **kwargs
):
    trainable = tune.with_parameters(
        hyperopt_with_tune,
        sdata=sdata,
        target_keys=target_keys,
        train_key=train_key,
        epochs=epochs,
        gpus=gpus,
        num_workers=num_workers,
        log_dir=log_dir,
        name=name,
        version=version,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        seq_transforms=seq_transforms,
        transform_kwargs=transform_kwargs,
        seed=seed,
        verbosity=verbosity,
        **kwargs
    )
    if scheduler_kwargs is None or len(scheduler_kwargs) == 0:
        scheduler_kwargs = default_scheduler_args[scheduler]
    scheduler = scheduler_dict[scheduler](**scheduler_kwargs)
    if algorithm_kwargs is None or len(algorithm_kwargs) == 0:
        algorithm_kwargs = default_algo_args[algorithm]
    algo = algo_dict[algorithm](**algorithm_kwargs)
    analysis = tune.run(
        trainable,
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        local_dir=settings.logging_dir,
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-val_loss",
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        name=name
    )

    return analysis