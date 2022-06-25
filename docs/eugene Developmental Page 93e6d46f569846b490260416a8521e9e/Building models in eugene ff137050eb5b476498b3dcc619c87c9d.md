# Building models in eugene

# All models are wrong, but some are useful

## [claim](https://www.notion.so/claim-3c878df2f69d40dca152d99a56ac900e) Modules

### `BasicFullyConnectedModule` —> `BasicFullyConnected`

### `BasicConv1D`

### `BasicRecurrent`

## LightningModules

## General

- Each module includes the following attributes
    - `strand`
    - `task`
    - `aggr`
    - `kwargs`
- Each module is command line runnable but can also be used with the **EUGENE API**
- Fix ts with hybrid to actually have reverse everything (not just convnet)

### `fcn.py`

- No capability for pwm interpetation
- Cannot be used to predict on variable genomic sequence

### `cnn.py`

- Cannot be used to predict on variable genomic sequence

### `rnn.py`

- No capability for pwm interpetation

### `hybrid.py`

## Model config (`/config/models`)

```bash
$STRAND$LIGHTNINGMODULE_$TASK.yaml
# Example: sshybrid_bin-clf_test.yaml
```