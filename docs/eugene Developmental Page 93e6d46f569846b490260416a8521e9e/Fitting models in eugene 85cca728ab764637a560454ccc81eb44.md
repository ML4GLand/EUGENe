# Fitting models in eugene

# “Its a little tight across the chest”

## Default training

- (built into [PyTorch Lightning](https://www.notion.so/PyTorch-Lightning-8c6fd6cfa6de4011a50902302b6836dc))
- Based on different experiments

## Fit trainer configs (`/config/trainers`)

### `generate_fit_config` function

```bash
$Descriptor_$method.yaml
# Example: benchmark2_fit.yaml
```

## `fit.sh` SLURM script

## Hyperoptimization (`/train/hyperopt.py`)

## Cross-validation(`/train/crossfold.py`)

- k-fold cross validation
    - [https://github.com/PyTorchLightning/pytorch-lightning/discussions/5820](https://github.com/PyTorchLightning/pytorch-lightning/discussions/5820)
    - [https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch](https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch)
    - https://github.com/PyTorchLightning/pytorch-lightning/issues/839
    - https://github.com/PyTorchLightning/pytorch-lightning/issues/1393
    - [https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py)
    - [https://stackoverflow.com/questions/69341479/how-to-use-cross-validation-in-pytorch-lightning](https://stackoverflow.com/questions/69341479/how-to-use-cross-validation-in-pytorch-lightning)
    - [https://devblog.pytorchlightning.ai/train-anything-with-lightning-custom-loops-4be32314c961](https://devblog.pytorchlightning.ai/train-anything-with-lightning-custom-loops-4be32314c961)
    - [https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py)

## Simplified python script (similar to `predict.py`) see below