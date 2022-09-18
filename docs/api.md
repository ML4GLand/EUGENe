```{eval-rst}
.. module:: eugene
```

```{eval-rst}
.. automodule:: eugene
   :noindex:
```

# API

Import EUGENe as:

```
import eugene as eu
```

```{note}
EUGENe is a package that is still active development, so you will likely run into errors and small typos if you choose to use hdWGCNA before its first stable release.
```

## Datasets

Every bioinformatician knows the pain of trying to track down and format a dataset for their needs. This module is designed to load several published datasets into SeqData objects with simple function calls. You can find a list of available datasets in the `datasets.csv` file or a live version here.

### Available datasets

You can get a list of available datasets returned as a `pandas.DataFrame` using the {func}`~eugene.datasets.get_dataset_info` function.

```{eval-rst}
.. module:: eugene.datasets
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   datasets.random1000
   datasets.ray13
   datasets.deBoer20
   datasets.jores21
   datasets.deAlmeida22
```

## Dataload `dl`

This module is designed to handle both the loading of data into Python objects and the compilation of those objects into Dlers for neural network training. This module is fundamental for the package and handles both the extraction and load aspects of the extract-transform-load steps.

### Input/Output (IO)

```{eval-rst}
.. module:: eugene.dl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   dl.read
   dl.read_csv
   dl.read_fasta
   dl.read_numpy
   dl.read_bed
   dl.read_bam
   dl.read_bigwig
   dl.read_h5sd

   dl.write
   dl.write_csv
   dl.write_fasta
   dl.write_numpy
   dl.write_h5sd
```

### SeqData

```{eval-rst}
.. module:: eugene.dl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   dl.SeqData
   dl.SeqData.write_h5sd
   dl.SeqData.to_dataset
   dl.concat
```

### SeqDataset

We need to be able to fluidly go between SeqData and PT datasets and DataLoaders. To do this we have implemented the SeqDataset class and attached a to_dataset() method to SeqData.

```{eval-rst}
.. module:: eugene.dl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   dl.SeqDataset
```

### Motif

```{eval-rst}
.. module:: eugene.dl.motif
.. currentmodule:: eugene.dl

.. autosummary::
   :toctree: api/

   motif.Motif
   motif.MinimalMEME
   motif.pwm_to_meme
   motif.filters_to_meme_sdata
   motif.get_jaspar_motifs
   motif.save_motifs_as_meme
   motif.load_meme
   motif.fimo_motifs
   motif.score_seqs
   motif.jaspar_annots_sdata
```

## Preprocess (`pp`)

This module is designed to let users interact and modify SeqData objects to prepare for model training and to allow for more in depth analysis. There are several classes of preprocessing functions that act on more familiar objects. These functions are agnostic of SeqData

### Sequence preprocessing

```{eval-rst}
.. module:: eugene.pp
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pp.sanitize_seq
   pp.sanitize_seqs
   pp.ascii_decode
   pp.ascii_encode
   pp.reverse_complement_seq
   pp.reverse_complement_seqs
   pp.ohe_seq
   pp.ohe_seqs
   pp.decode_seq
   pp.decode_seqs
   pp.dinuc_shuffle_seq
   pp.dinuc_shuffle_seqs
   pp.perturb_seq
   pp.perturb_seqs
   pp.feature_implant_seq
   pp.feature_implant_across_seq
```

### Dataset preprocessing

```{eval-rst}
.. module:: eugene.pp
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pp.split_train_test
   pp.standardize_features
   pp.binarize_values
```

### SeqData preprocessing

```{eval-rst}
.. module:: eugene.pp
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pp.sanitize_seqs_sdata
   pp.ohe_seqs_sdata
   pp.reverse_complement_seqs_sdata
   pp.clean_nan_targets_sdata
   pp.clamp_targets_sdata
   pp.scale_targets_sdata
   pp.binarize_targets_sdata
   pp.train_test_split_sdata
   pp.add_ranges_pos_annot
   pp.prepare_seqs_sdata
```

## Models

This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences
We specify three main classes of model: base_model, sota_model, and custom_model

### `BaseModels`

```{eval-rst}
.. module:: eugene.models
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   models.FCN
   models.CNN
   models.RNN
   models.Hybrid
```

### SOTA Models

```{eval-rst}
.. module:: eugene.models
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   models.DeepBind
   models.DeepSEA
```

### Custom Models

```{eval-rst}
.. module:: eugene.models
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   models.Jores21CNN
   models.Kopp21CNN
```

### Initialization

```{eval-rst}
.. module:: eugene.models
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   models.init_weights
   models.init_from_motifs
```

## Training

Similarly to prediction, I feel like PL Trainers will take care of most of this. But I guess their are some considerations listed below

### Basic Training

```{eval-rst}
.. module:: eugene.train
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   train.fit
```

## Evaluate

### Predictions

```{eval-rst}
.. module:: eugene.evaluate
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   evaluate.predictions
   evaluate.train_val_predictions
```

### Metrics

```{eval-rst}
.. module:: eugene.evaluate
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   evaluate.median_calc
   evaluate.auc_calc
   evaluate.escore
   evaluate.rnacomplete_metrics
   evaluate.rnacomplete_metrics_sdata_plot
   evaluate.rnacomplete_metrics_sdata_table
```

## Interpret

### Filter visualization

```{eval-rst}
.. module:: eugene.interpret
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   interpret.generate_pfms_sdata
```

### Feature attributions

```{eval-rst}
.. module:: eugene.interpret
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   interpret.nn_explain
   interpret.feature_attribution_sdata
   interpret.aggregate_importances_sdata
```

### *In silico*

```{eval-rst}
.. module:: eugene.interpret
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   interpret.best_k_muts
   interpret.best_mut_seqs
   interpret.evolution
   interpret.evolve_seqs_sdata
   interpret.feature_implant_seq_sdata
   interpret.feature_implant_seqs_sdata
```

### Dimensionality reduction

```{eval-rst}
.. module:: eugene.interpret
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   interpret.pca
   interpret.umap
```

## Plotting

### Categorical plotting

```{eval-rst}
.. module:: eugene.pl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pl.countplot
   pl.histplot
   pl.boxplot
   pl.violinplot
   pl.scatterplot
```

### Training summaries

```{eval-rst}
.. module:: eugene.pl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pl.metric_curve
   pl.loss_curve
   pl.training_summary
```

### Performance

```{eval-rst}
.. module:: eugene.pl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pl.performance_scatter
   pl.confusion_mtx
   pl.auroc
   pl.auprc
   pl.performance_summary

```

### Sequences

```{eval-rst}
.. module:: eugene.pl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pl.seq_track
   pl.multiseq_track
   pl.lm_seq_track
   pl.lm_multiseq_track
   pl.filter_viz
   pl.lm_filter_viz
   pl.lm_multifilter_viz
   pl.feature_implant_plot
```

### Dimensionality reduction

```{eval-rst}
.. module:: eugene.pl
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   pl.pca
   pl.umap
   pl.skree
```

## Utilities

### Random sequence generation

```{eval-rst}
.. module:: eugene.utils
.. currentmodule:: eugene

.. autosummary::
   :toctree: api/

   utils.random_base
   utils.random_seq
   utils.random_seqs
   utils.generate_random_data
```
