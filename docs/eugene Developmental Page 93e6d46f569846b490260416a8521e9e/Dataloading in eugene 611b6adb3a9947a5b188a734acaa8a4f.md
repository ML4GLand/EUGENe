# Dataloading in eugene

# “Never argue with the data”

# EUGENE NEEDS A DATASTRUCTURE

## Dataloading (`/dataloading`)

### `load_data.py` —> `io.py`

- `load_csv()`: implemented
- `load_fasta()`: implemented
- `load_numpy()`: implemented
- `load()`: wraps the above —> implemented

## Dataloaders (`/dataloaders`)

Use anndata to create a object to work with?

### `SeqDataset`

- Minimally requires iterable of sequences

### `SeqDataloader`

### `SeqDataModule`

- Builds a `SeqDataset` object from a file or set of files
- Composes passed in transforms (expects them to be of class transforms)
- Freakin split finally works:
    
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1565
    

### `AnnDataset`?

## Data config (`_generate_data_config.py`)

- `generate_data_config()`

## Standardize the naming and include all in same directory

```bash
$DATASET_$FORMAT_$TASK_$METHOD.yaml
# Example: 2021_OLS_Library_All_OHE-T_bin-clf_test.yaml
```