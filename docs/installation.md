# Installation
EUGENe is a Python package, and can be installed using `pip`:

```bash
pip install 'eugene-tools'
```

The extras `[janggu,kipoi,memesuite]` install dependencies that are needed to use functions from the [Janggu](https://janggu.readthedocs.io), [Kipoi](https://kipoi.org/) and [MEME](https://meme-suite.org/meme) suite respectively. You can install them using:

```bash
pip install 'eugene-tools[janggu,kipoi,memesuite]'
```

Or if you only want a single extra:

```bash
pip install 'eugene-tools[janggu]'
```

You can run a quick test of your installation, by opening an interactive Python prompt and running

```python
import eugene as eu
```

```{note}
In it's current state, EUGENe is a pretty dependency heavy package. Some of this is due to the nature of the field, but we are also working on slimming down the package for future releases
```

## Development Version
To work with the latest version [on GitHub](https://github.com/cartercompbio/EUGENe), clone the repository and `cd` into its root directory.

```bash
git clone https://github.com/cartercompbio/EUGENe.git 
cd EUGENe
```

Then, install the package in development mode:

```bash
pip install -e .[dev]
```

```{note}
If you want to edit the docs, you will need top run `pip install -e .[docs]
```

## Troubleshooting
If you have any issues installing, please [open an issue](https://github.com/cartercompbio/EUGENe/issues) on GitHub! We will do our best to help you ASAP!
