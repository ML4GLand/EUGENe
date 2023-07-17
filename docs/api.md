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

## `models`

```
from eugene import models
```

This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences.

### Blocks
Blocks are composed to create architectures in EUGENe. You can find all the arguments that would be passed into the `dense_kwargs` and `recurrent_kwargs` arguments of all built-in model in the `DenseBlock` and `RecurrentBlock` classes, respectively. See the [towers section](#towers) for more information on the `conv_kwargs` argument.

```{eval-rst}

```{eval-rst}
.. module:: eugene.models
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.DenseBlock
   models.Conv1DBlock
   models.RecurrentBlock
```

### Towers
The `Conv1DTower` class is currently used for all built-in CNNs. This will be deprecated in the future in favor of the more general `Tower` class. For now, you can find all the arguments that would be passed into the `cnn_kwargs` argument of all built-in CNNs in the `Conv1DTower` class.

```{eval-rst}
.. module:: eugene.models
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.Tower
   models.Conv1DTower
```

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
Arguments for the `cnn_kwargs`, `recurrent_kwargs` and `dense_kwargs` of all models can be found in the `Conv1DTower`, `RecurrentBlock` and `DenseBlock` classes, respectively. See the [blocks section](#blocks) and the [towers section](#towers) for more information. The `Satori` architecture currently uses the `MultiHeadAttention` layer which can be found at `eugene.models.base._layers` for more information on the `mha_kwargs` argument.

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

## `train`

```
from eugene import train
```

Training procedures for data and models.

```{eval-rst}
.. module:: eugene.train
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/

   train.fit
   train.fit_sequence_module
   train.hyperopt
```

## `evaluate`

```
from eugene import evaluate
```

Evaluation functions for trained models. Both prediction helpers and metrics.

```{eval-rst}
.. module:: eugene.evaluate
```

```{eval-rst}
.. currentmodule:: eugene
```

### Predictions

```{eval-rst}
.. autosummary::
   :toctree: api/

   evaluate.predictions
   evaluate.predictions_sequence_module
   evaluate.train_val_predictions
   evaluate.train_val_predictions_sequence_module
```

## `interpret`

```
from eugene import interpret
```

Interpretation suite of EUGENe, currently broken into filter visualization, feature attribution and *in silico* experimentation

```{eval-rst}
.. module:: eugene.intepret
```

```{eval-rst}
.. currentmodule:: eugene
```

### Filter interpretation

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.generate_pfms_sdata
   interpret.filters_to_meme_sdata
```

### Attribution analysis

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.attribute_sdata
```

### *Global importance analysis (GIA)*

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.positional_gia_sdata
   interpret.motif_distance_dependence_gia
```

### Generative

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.evolve_seqs_sdata
```

## `plot`

```
from eugene import plot
```

Plotting suite in EUGENe for multiple aspects of the workflow.

```{eval-rst}
.. module:: eugene.plot
```

```{eval-rst}
.. currentmodule:: eugene
```

### Categorical plotting

```{eval-rst}
.. autosummary::
   :toctree: api/

   plot.countplot
   plot.histplot
   plot.boxplot
   plot.violinplot
   plot.scatterplot
```

### Training summaries

```{eval-rst}
.. autosummary::
   :toctree: api/

   plot.metric_curve
   plot.loss_curve
   plot.training_summary
```

### Performance

```{eval-rst}
.. autosummary::
   :toctree: api/

   plot.performance_scatter
   plot.confusion_mtx
   plot.auroc
   plot.auprc
   plot.performance_summary
```

### Sequences

```{eval-rst}
.. autosummary::
   :toctree: api/

   plot.seq_track
   plot.multiseq_track
   plot.filter_viz
   plot.multifilter_viz
```

### Global importance analysis (GIA)

```{eval-rst}
.. autosummary::
   :toctree: api/

   plot.positional_gia_plot
   plot.distance_cooperativity_gia_plot
```

## Utilities

```{eval-rst}
.. module:: eugene.utils
```

```{eval-rst}
.. currentmodule:: eugene
```

### File I/O

```{eval-rst}
.. autosummary::
   :toctree: api/

   utils.make_dirs
```
