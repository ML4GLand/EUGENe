# Preprocessing data with eugene

## Data preprocessing (`/preprocess`)

### Sequence preprocessing

- `_dataset_preprocess.py`
    - `split_train_test()`
    - `standardize_features()`
- `_encoding.py`
    - `oheDNA()`
    - `encodeDNA()`
    - `decodeDNA()`
    - `ascii_encode()`
    - `ascii_decode()`
- `_transforms.py`
    - `ReverseComplement`
    - `Augment`
    - `OneHotEncode`
    - `ToTensor`
- `_feature_selection.py`
- `_utils.py`
    - Process sequences into different formats
        - `_get_index_dict()`
        - …
        - `pad_sequences()`
    - Generate random sequences
        - `random_base()`
        - …
    - Shuffle dinucleotides
        - `dinuc_shuffle()`

### Example notebooks for preprocessing data from different formats

- [ ]  `Preprocess_From_TSV.ipynb`

### Example scripts and code for downloading commonly used data from other sources
