# Module Usage Principles
This page is dedicated to giving users a deep dive into the modules that make up EUGENe and usage principles for each one. We will first briefly describe the workflow of EUGENe (practically illustrated in the {doc}`basic usage worfklow <basic_usage_tutorial>` tutorial), then go into each stage of the workflow in detail.

## Workflow
A EUGENe workflow consists of three major stages that themselves can be broken down into several substeps. These are:

1. Extract, transform and load (ETL) datasets for deep learning
2. Instantiate, initialize and train (IIT) deep learning models with PyTorch Lightning
3. Evaluate and interpret (EI) trained models with a variety of methods and visualizations

## Extract, Transform, Load (ETL)

### `SeqData` -- The core data container of EUGENe
Our current release of EUGENe relies on several subpackages. `SeqData` is...

### `SeqDatasets` -- Quickly start your development or benchmarking
Every bioinformatician knows the pain of trying to track down and format a dataset for their needs. This module is meant to ease that burden. It also sets up users to quickly benchmark methods and rapidly prototype ideas! We designed the datasets module with the following principles in mind:

1. A file containing a list of datasets and their descriptions is kept in `datasets.csv` that can be accessed with the `eu.dl.get_dataset_info()` function. You can also check out the [datasets] API for a list of currently available datasets and their descriptions.

2. Datasets are returned to users as `SeqData` objects with simple calls (e.g. eu.datasets.dataset_name()).

3. If the user does not have the dataset downloaded in the location specified by the command, EUGENe works to download it for you.

4. EUGENe installations come with a single preloaded dataset (random1000) representing random sequences and targets. These are designed for testing purposes.

5. Adding datasets is a pretty straightforward process! We have developed a tutorial notebook that walks you through the process of adding a dataset to EUGENe. You can find it [here](https://github.com/cartercompbio/EUGENe/blob/main/tutorials/adding_a_dataset_tutorial.ipynb). We strongly encourage users to do so and submit pull requests for them.

Check out the [`datasets` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.datasets) for a list of currently available datasets and their descriptions.

### `preprocess` -- Prepare data for sequence-based deep learning
This module is designed to let users interact and modify `SeqData` objects to prepare for model training and to allow for more in depth analysis.

1. There are several classes of preprocessing functions that act on more familiar objects. Ideally, these functions are agnostic of `SeqData`.

2. Sequence preprocessing (`eu.pp.*_seq()` and `eu.pp.*_seqs()`) functions act on sequence. Ideally, each type of function (reverse complement, one-hot encode etc.) should have a single sequence function and a multiple sequence function.

3. Ideally, each multiple sequence function should be parallelizable (or vectorized) and should not just loop through the sequences.

4. Dataset preprocessing functions are meant to serve as helpers to perform the more “traditional” machine learning preprocessing steps (e.g. train/test split, feature standardization etc.).

5. All preprocess functions should ideally have `SeqData` wrappers (`eu.pp.*_seq_sdata()`).

6. By default, `SeqData` objects are modified in place, but if `copy = True` is specified a copy is returned.

7. Adding a preprocessing function is a simple process.

Check out the [`preprocess` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.pp) for a list of preprocessing functions.

### `dataload` -- Read/write data from many common file formats
This module is designed to handle both the loading of data into Python objects and the compilation of those objects into dataloaders for neural network training.

1. We want to be able to generalize loading data from csv, numpy, fasta, and h5sd (see below) into a `SeqData` object (see below).

2. `SeqData` objects are the core data containers of EUGENe.

3. We need to be able to fluidly go between `SeqData` and PyTorch datasets and DataLoaders.

4. Normally, `SeqData` objects should be saved as `h5sd` files.

5. We wrap Janggu functions for reading from bed, bigWig, and bam.

Check out the [`dataload` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.dl) for a list of dataloaders and dataloading functions.

## Instantiate, Initialize, Train (IIT)

### `models` -- Instantiate and initialize neural network architectures
This module is designed to allow users to easily build and initialize several neural network architectures that are designed for biological sequences.

1. Fundamentally, a model needs to be a PyTorch module with the `init` and `forward` functions implemented.

2. Every model should ideally extend the `BaseModel` class that is implemented in the `eugene/models/base/_base_model.py` file.

3. By default we assume a single stranded (ss), regression model (regression) that is trained to optimize mean squared error (mse).

4. We specify three main classes of model: Base Model, SOTA Model, and Custom Model.

5. Each model is built from a combination of modules and standard PyTorch code.

6. Models are either instantiated through calls to `eu.models.ModelName()` constructors or through configuration files.

7. After model instantiation, users can edit things like optimizers and loss functions, but cannot edit model architecture!

Check out the [`models` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.models) for a list of models and model building functions.

### `train` -- Fit parameters to your data
For basic trianing, this module is mainly a wrapper around [PyTorch Lightning's trainers](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html). In future releases, we will add more advanced training functionality for doing things like hyperparameter optimization, GAN training, and more.

1. Logging is handled by PyTorch Lightning via the [Tensorboard logger](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.TensorBoardLogger.html#pytorch_lightning.loggers.TensorBoardLogger).

2. Metric tracking is handled by [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/).

3. Loss functions, optimizers and learning rate schedulers can be instantiated with the model, or assigned after instantiation

Check out the [`train` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.train) for more info on the current fit function.

## Evaluate and Interpret (EI)

### `evaluate` -- Validate and explore models on new data
Similarly to training, this module is mainly a wrapper around [PyTorch Lightning's trainers](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html). However, we have also have begun a metrics library to help users calculate training metrics on their `SeqData` objects (more details coming soon!).

1. Predictions on a `SeqData` are saved in `seqs_annot` attribute and to disk by default (with column names and file paths specified by the user)

Check out the [`evaluate` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.evaluate) for more info on the current evaluate functions, including how to save train and validation set prediction ouptuts separately.

### `interpret` -- Investigate learned model behavior
There is no shortage of ways one could come up with to try to interpret a trained model. We have included three core intepretation categories in EUGENe so far:

1. Visualize filters as PWMs

2. Calculate per nucleotide importance scores

3. Use the model as an in silico oracle to predict the effect of mutations and other perturbations

Check out the [`interpret` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.interpret) for more info on the current interpret functions available.

### `plot` -- Visualize it all as you go
This module is designed to help users visualize their data and results from a workflow. We have included several plotting functions that are designed to work with `SeqData` objects and can be broken up into several different categories.

1. Plotting functions call Matplotlib and Seaborn functions under the hood and act primarily on `SeqData` objects.

2. [Categorical plotting](https://eugene-tools.readthedocs.io/en/latest/api.html#categorical-plotting) functions can be used for exploratory data analysis (EDA) and are designed to work with information in the `seqs_annot` attribute of `SeqData` objects.

3. [Training summary](https://eugene-tools.readthedocs.io/en/latest/api.html#training-summaries) functions are designed to help users visualize the training process and act on Tensorboard events files.

4. [Performance plotting](https://eugene-tools.readthedocs.io/en/latest/api.html#performance) functions are designed to help users visualize the performance of their models data in a `SeqData` object. 

5. [Sequence plotting](https://eugene-tools.readthedocs.io/en/latest/api.html#sequences) functions are designed to help users visualize the sequences in their `SeqData` objects and any annotations that are present for those sequences. These annotations might include feature attributions, predicted TF binding sites, or other information.

6. [Dimensionality reduction plotting](https://eugene-tools.readthedocs.io/en/latest/api.html#id1) functions are designed to help users visualize the results of dimensionality reduction techniques performed on `SeqData` objects.

Check out the [`plot` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.plot) for more info on the current plot functions available.

## `utils` -- Miscellaneous utilities
This module is designed to be a catch all for functions that don't fit into the other modules. This includes helper functions for things like generating random sequences, tracking `SeqData` objects, and more.

Check out the [`utils` API](https://eugene-tools.readthedocs.io/en/latest/api.html#module-eugene.utils) for more info on the current utils functions available.
