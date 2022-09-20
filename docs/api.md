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
EUGENe is a package that is still active development, so there's a good chance you'll hit an error to if you use EUGENe before its first stable release.
```

## Datasets

```{eval-rst}
.. module:: eugene.datasets
```

```{eval-rst}
.. currentmodule:: eugene
```

### Available datasets

You can get a list of available datasets returned as a `pandas.DataFrame` using the {func}`~eugene.datasets.get_dataset_info` function.

```{eval-rst}
.. autosummary::
   :toctree: api/

   datasets.random1000
   datasets.ray13
   datasets.deBoer20
   datasets.jores21
   datasets.deAlmeida22
```

## Dataload `dl`

```{eval-rst}
.. module:: eugene.dl
```

```{eval-rst}
.. currentmodule:: eugene
```

### Input/Output (IO)

The (`io`) functions handle reading and writing from and to files on disk.

```{eval-rst}
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
These are the few core functions you can call on SeqData objects

```{eval-rst}
.. autosummary::
   :toctree: api/

   dl.SeqData
   dl.SeqData.write_h5sd
   dl.SeqData.to_dataset
   dl.concat
```

### SeqDataset

We need to be able to fluidly go between SeqData and PyTorch datasets and DataLoaders. To do this we have implemented the SeqDataset class and attached a to_dataset() method to SeqData.

```{eval-rst}
.. autosummary::
   :toctree: api/

   dl.SeqDataset
```

### Motif

These functions are for working with MEME format. They are used to read in MEME files and convert them to SeqData objects.

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

```{eval-rst}
.. module:: eugene.pp
```

```{eval-rst}
.. currentmodule:: eugene
```

This module is designed to let users interact and modify SeqData objects to prepare for model training and to allow for more in depth analysis. There are several classes of preprocessing functions that act on more familiar objects. These functions are agnostic of SeqData

### Sequence preprocessing

```{eval-rst}
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
.. autosummary::
   :toctree: api/

   pp.split_train_test
   pp.standardize_features
   pp.binarize_values
```

### SeqData preprocessing

```{eval-rst}
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

```{eval-rst}
.. module:: eugene.models
```

```{eval-rst}
.. currentmodule:: eugene
```

This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences
We specify three main classes of model: base_model, sota_model, and custom_model

### `BaseModels`

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.FCN
   models.CNN
   models.RNN
   models.Hybrid
```

### SOTA Models

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.DeepBind
   models.DeepSEA
```

### Custom Models

```{eval-rst}
.. autosummary::
   :toctree: api/classes

   models.Jores21CNN
   models.Kopp21CNN
```

### Initialization

```{eval-rst}
.. autosummary::
   :toctree: api/

   models.init_weights
   models.init_from_motifs
```

## Training

```{eval-rst}
.. module:: eugene.train
```

```{eval-rst}
.. currentmodule:: eugene
```

Similarly to prediction, I feel like PL Trainers will take care of most of this. But I guess their are some considerations listed below

### Basic Training

```{eval-rst}
.. autosummary::
   :toctree: api/

   train.fit
```

## Evaluate

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
   evaluate.train_val_predictions
```

### Metrics

```{eval-rst}
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

```{eval-rst}
.. module:: eugene.intepret
```

```{eval-rst}
.. currentmodule:: eugene
```

### Filter visualization

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.generate_pfms_sdata
```

### Feature attributions

```{eval-rst}
.. autosummary::
   :toctree: api/

   interpret.nn_explain
   interpret.feature_attribution_sdata
   interpret.aggregate_importances_sdata
```

### *In silico*

```{eval-rst}
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
.. autosummary::
   :toctree: api/

   interpret.pca
   interpret.umap
```

## Plotting

```{eval-rst}
.. module:: eugene.pl
```

```{eval-rst}
.. currentmodule:: eugene
```

### Categorical plotting

```{eval-rst}
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
.. autosummary::
   :toctree: api/

   pl.metric_curve
   pl.loss_curve
   pl.training_summary
```

### Performance

```{eval-rst}
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
.. autosummary::
   :toctree: api/

   pl.pca
   pl.umap
   pl.skree
```

## Utilities

```{eval-rst}
.. module:: eugene.utils
```

```{eval-rst}
.. currentmodule:: eugene
```

### Random sequence generation

```{eval-rst}
.. autosummary::
   :toctree: api/

   utils.random_base
   utils.random_seq
   utils.random_seqs
   utils.generate_random_data
```
