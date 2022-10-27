[![PyPI version](https://badge.fury.io/py/eugene-tools.svg)](https://badge.fury.io/py/eugene-tools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eugene-tools)

<img src="_static/EugeneLogoText.png" alt="EUGENe Logo" width=600>

# Elucidating the Utility of Genomic Elements with Neuralnets

<img src="_static/workflow.png" alt="EUGENe workflow" width=600>

EUGENe represents a computational framework for machine learning based modeling of regulatory sequences. It is designed after the [Scanpy](https://scanpy.readthedocs.io/en/stable/) package for single cell analysis in Python and is meant to make the development of deep learning worlflows in the genomics field more findable, accessible, interoperitable and reproducible (FAIR). 

EUGENe consists of several modules for handling data and for building, training, evaluating and interpreting deep learners that predict annotations of biological sequences. EUGENe is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. [Jupyter](https://jupyter.org/)).

* Get started by installing EUGENe {doc}`installation <installation>`
* Check out the {doc}`basic usage worfklow <basic_usage_tutorial>` tutorial for an example of how to run a EUGENe workflow
* For a more in depth look at EUGENe, browse the main {doc}`API <api>` and read through the {doc}`usage principles <usage-principles>`

```{note}
EUGENe is a package that is still under active development, so there's bound to be some rough edges to smooth out. However, catching errors, typos, etc. is incredbily valuable for us! If you run into such errors or have any questions, please open an issue!
```

If you use EUGENe for your research, please cite our preprint: [Klie *et al.* bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.10.24.513593v1)

```{toctree}
:hidden: true
:maxdepth: 1

installation
basic_usage_tutorial
api
usage-principles
standards
contributors
references
