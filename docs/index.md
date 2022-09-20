[![PyPI version](https://badge.fury.io/py/eugene-tools.svg)](https://badge.fury.io/py/eugene-tools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eugene-tools)

# EUGENe (Elucidating the Utility of Genomic Elements with Neuralnets)
<img src="_static/workflow.png" alt="EUGENe workflow" width=800>

EUGENe represents a computational framework for machine learning based modeling of regulatory sequences. It is designed after the [Scanpy](https://scanpy.readthedocs.io/en/stable/) package for single cell analysis in Python and is meant to make the development of deep learning worlflows in the genomics field more findable, accessible, interoperitable and reproducible (FAIR). EUGENe consists of several modules for handling data and for building, training, evaluating and interpreting deep learners that predict annotations of biological sequences. EUGENe is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. Jupyter).

* Get started by installing EUGENe {doc}`installation <installation>`
* Check out the {doc}`basic_usage_worfklow <basic_usage_worfklow>` for an example of how to use EUGENe
* For a more in depth look at EUGENe, browse the main {doc}`API <api>` and read through the {doc}`usage principles <usage-principles>`

```{note}
EUGENe is a package that is still under active development, so there's a good chance you'll hit an error to if you use EUGENe before its first stable release. However, catching these errors is incredbily valuable for us! If you run into such errors or have any questions, please open an issue!
```

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
