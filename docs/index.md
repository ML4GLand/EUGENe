[![PyPI version](https://badge.fury.io/py/eugene-tools.svg)](https://badge.fury.io/py/eugene-tools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eugene-tools)

<img src="_static/eugene_logo.png" alt="EUGENe Logo" width=600>

# Elucidating the Utility of Genomic Elements with Neuralnets

EUGENe represents a computational toolkit written in Python for building sequence-based deep learning models in genomics. EUGENe provides a unified interface for handling data and for building, training, evaluating and interpreting deep learners that predict annotations of biological sequences. Examples of ways EUGENe can be used include:

* **Learn the fundamentals through practice.** Architect, train and interpret deep learning models using EUGENe's streamlined workflow. See if you can recapitulate the results of published studies!
* **Torture existing models.** Find failure modes or novel behaviors of existing models on novel or synthetically designed datasets.
* **Apply existing architectures to new data.** Have a new dataset but not sure how to build a model for it? Try training and interpreting established architectures to see what they learn.
* **Build new architecturess.** Benchmark their performance and interpretability against existing architectures on established datasets

 EUGENe is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. [Jupyter](https://jupyter.org/)).

# Getting started
* {doc}`Install EUGENe <installation>`
* (Optional) Read through the {doc}`usage principles <usage-principles>` to get a better understanding of how EUGENe works in practice
* Check out the {doc}`basic usage tutorial <basic_usage_tutorial>` for an example of how to run an end-to-end EUGENe workflow
* Browse the main {doc}`API <api>`  page to see all the functionality that EUGENe provides

```{note}
EUGENe is a package that is under active development. If you run errors, please open an issue on the EUGENe GitHub. Your feedback is incredbily valuable for us!
```

If you use EUGENe for your research, please cite our preprint: [Klie *et al.* bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.10.24.513593v1)

# Contributing
EUGENe is an open-source project and we welcome contributions from the community. If you are interested in contributing, please see the {doc}`contributor's guide <contributors>`.

```{toctree}
:hidden: true
:maxdepth: 1

installation
usage-principles
basic_usage_tutorial
api
standards
contributors
references
