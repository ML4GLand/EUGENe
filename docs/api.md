```{eval-rst}
.. module:: eugene
```

```{eval-rst}
.. automodule:: eugene
   :noindex:
```

# API

## `preprocess`

```
from eugene import preprocess
```

This module is designed to let users interact and modify SeqData objects to prepare for model training and other steps of the workflow. There are three main classes of preprocessing functions.

```{eval-rst}
.. module:: eugene.preprocess
```

```{eval-rst}
.. currentmodule:: eugene
```

### Sequence preprocessing

```{eval-rst}
.. autosummary::
   :toctree: api/

   preprocess.make_unique_ids_sdata
   preprocess.pad_seqs_sdata
   preprocess.ohe_seqs_sdata
```

### Train-test splitting

```{eval-rst}
.. autosummary::
   :toctree: api/
   
   preprocess.train_test_chrom_split
   preprocess.train_test_homology_split
   preprocess.train_test_random_split
```

### Target preprocessing

```{eval-rst}
.. autosummary::
   :toctree: api/

   preprocess.clamp_targets_sdata
   preprocess.scale_targets_sdata
```

## `dataload`

```
from eugene import dataload
```

This module is designed to help users prepare their SeqDatas for model training and other steps of the workflow (e.g. augmentation)

```{eval-rst}
.. module:: eugene.dataload
```

```{eval-rst}
.. currentmodule:: eugene
```

### SeqData utilities

```{eval-rst}
.. autosummary::
   :toctree: api/

   dataload.concat_sdatas
   dataload.add_obs
```

### Augmentation

```{eval-rst}
.. autosummary::
   :toctree: api/

   dataload.RandomRC
```
```{eval-rst}
.. module:: eugene
```

```{eval-rst}
.. automodule:: eugene
   :noindex:
```

## `models`

```
from eugene import models
```

This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences.

### LightningModules

```{eval-rst}
.. module:: eugene.models
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.SequenceModule
   models.ProfileModule
```

### Initialization

```{eval-rst}
.. autosummary::
   :toctree: api/

   models.init_weights
   models.init_motif_weights
```

### Zoo

```{eval-rst}
.. module:: eugene.models.zoo
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.zoo.FCN
   models.zoo.dsFCN
   models.zoo.CNN
   models.zoo.dsCNN
   models.zoo.RNN
   models.zoo.dsRNN
   models.zoo.Hybrid
   models.zoo.dsHybrid
   models.zoo.TutorialCNN
   models.zoo.DeepBind
   models.zoo.ResidualBind
   models.zoo.Kopp21CNN
   models.zoo.DeepSEA
   models.zoo.Basset
   models.zoo.FactorizedBasset
   models.zoo.DanQ
   models.zoo.Satori
   models.zoo.Jores21CNN
   models.zoo.DeepSTARR
   models.zoo.BPNet
```

### Utilities

```{eval-rst}
.. autosummary::
   :toctree: api/

   models.list_available_layers
   models.get_layer
   models.load_config
```
