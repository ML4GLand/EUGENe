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

## Datasets

```{eval-rst}
.. module:: eugene.datasets
```

```{eval-rst}
.. currentmodule:: eugene
```

### Available datasets
You can get a list of available datasets returned as a `pandas.DataFrame` using the `eugene.datasets.get_dataset_info()` function.

```{eval-rst}
.. autosummary::
   :toctree: api/
    
   datasets.get_dataset_info
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
.. module:: eugene.dataload.dataloaders
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/

   dataload.dataloaders.SeqData
   dataload.dataloaders.SeqData.write_h5sd
   dataload.dataloaders.SeqData.to_dataset
   dataload.concat
```

### SeqDataset
We often need to be able to fluidly go between SeqData and PyTorch datasets and DataLoaders. To do this we have implemented the `SeqDataset` class and attached a `to_dataset()` method to SeqData.

```{eval-rst}
.. autosummary::
   :toctree: api/

   dataload.dataloaders.SeqDataset
   dataload.dataloaders.SeqDataset.to_dataloader
```

### Motif
These functions are for working with MEME format. They are used to read in MEME files and convert them to SeqData objects.

```{eval-rst}
.. module:: eugene.dataload.motif
```

```{eval-rst}
.. currentmodule:: eugene
```

```{eval-rst}
.. autosummary::
   :toctree: api/

   dataload.motif.Motif
   dataload.motif.MinimalMEME
   dataload.motif.pwm_to_meme
   dataload.motif.filters_to_meme_sdata
   dataload.motif.get_jaspar_motifs
   dataload.motif.save_motifs_as_meme
   dataload.motif.load_meme
   dataload.motif.fimo_motifs
   dataload.motif.score_seqs
   dataload.motif.jaspar_annots_sdata
```

## Preprocess (`pp`)
This module is designed to let users interact and modify SeqData objects to prepare for model training and other steps of the workflow. There are three main classes of preprocessing functions.

```{eval-rst}
.. module:: eugene.pp
```

```{eval-rst}
.. currentmodule:: eugene
```

### Sequence preprocessing

```{eval-rst}
.. autosummary::
   :toctree: api/

   pp.sanitize_seq
   pp.sanitize_seqs
   pp.ascii_encode_seq
   pp.ascii_encode_seqs
   pp.ascii_decode_seq
   pp.ascii_decode_seqs
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
   pp.add_ranges_sdata
   pp.prepare_seqs_sdata
```

## Models
This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences. We specify three main classes of model: Base Models, SOTA Models, and Custom Models.

```{eval-rst}
.. module:: eugene.models
```

```{eval-rst}
.. currentmodule:: eugene
```

### Base Models

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

   models.TutorialCNN
   models.Jores21CNN
   models.Kopp21CNN
```

### Initialization

```{eval-rst}
.. autosummary::
   :toctree: api/

   models.load_config
   models.init_weights
   models.init_from_motifs
```

## Training
Training procedures for data and models.

```{eval-rst}
.. module:: eugene.train
```

```{eval-rst}
.. currentmodule:: eugene
```

### Basic Training

```{eval-rst}
.. autosummary::
   :toctree: api/

   train.fit
```

## Evaluate
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
Interpretation suite of EUGENe, currently broken into filter visualization, feature attribution and *in silico* experimentation
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
Plotting suite in EUGENe for multiple aspects of the workflow.

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

   pl.seq_track_features
   pl.multiseq_track_features
   pl.seq_track
   pl.multiseq_track
   pl.filter_viz_seqlogo
   pl.filter_viz
   pl.multifilter_viz
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
