EUGENe Overview
================

What is EUGENe?
----------------
EUGENe represents a computational framework for building models of regulatory sequences as input. It is designed after the `scverse <https://scverse.org/>`_ framework for single cell analysis in Python and is meant to make the development in the deep learning genomics field more accessible. EUGENe consists of a codebase for building, training, validating and interpreting several deep learners that model sequence-based data. EUGENe is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. `Jupyter <https://jupyter.org/>`_), however we have also implemented several key functions via the command line.

What does EUGENe do?
--------------------
- Models will take DNA input and derivatives. But it is worth noting that extensions to other sequence based inputs are not out of the question
- Most of the tasks performed by this framework will be supervised (i.e., labels are necessary), though we are looking to extend into some more unsupervised, semi-supervised and probabilistic modeling techniques
- EUGENE is primarily designed for predicting tissue-specific enhancer activity from massively parallel reporter assay (MPRA) libraries (or STARR-seq), but is general enough to be applied to a variety of sequence-based prediction tasks.
- Almost everything operates on either a :doc:`../api/dataloading/dataloaders/SeqData` object, a Model, or a combination of the two.
- Models in EUGENe are built off of `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ for ease of use.

How is EUGENe used?
--------------------

#. Read data into a :doc:`../api/dataloading/dataloaders/SeqData` object.

    * A variety of datatypes such as ``csv``, ```numpy``, ``fasta``, etc, can be used.

#. Preprocess the :doc:`../api/dataloading/dataloaders/SeqData` object for training.

    * Add one hot encoding (``one_hot_encode_data()``), reverse complements (``reverse_complement_data()``), and training indices (``test_train_split_data()``).
    * The ``prepare_data()`` function can be used to automate these steps.

#. Initialize a model.

    * One of many core models can be used such as ``FCN``, ``CNN``, ``DeepBind``, etc.
    * Custom models can be made by extending ``BaseModel``.
    * Hyperparameters can be defined at the definition of a model.

#. Train the model.

    * The ``fit()`` function is used to train the model using an :doc:`../api/dataloading/dataloaders/SeqData` object.

#. Predict on a similar sequence.

    * The ``train_val_predictions()`` and ``predictions()`` functions can store predicted values into the :doc:`../api/dataloading/dataloaders/SeqData` object.

#. Interpret the model

    * Perform feature attribution or PWM visualization to gauge what the model has learned.
