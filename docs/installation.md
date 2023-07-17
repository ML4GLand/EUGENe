# Installation
EUGENe is a Python package, and can be installed using `pip`:

```bash
pip install 'eugene-tools'
```

We highly recommend using a virtual environment to install EUGENe to avoid conflicting dependencies with other packages. If you are unfamiliar with virtual environments, we recommend using Miniconda.

```bash
VERSION=3.9 # or 3.8-3.12
conda create -n eugene python=$VERSION
```

We also recommend installing mamba to speed up the installation process.

# Developmental installation
To work with the latest version [on GitHub](https://github.com/ML4GLand/EUGENe), clone the repository and `cd` into its root directory.

```bash
git clone https://github.com/cartercompbio/EUGENe.git
cd EUGENe
```

Then, install the package in development mode:

```bash
pip install -e .
```

Extras for development `[dev, docs]` can be installed using:

```bash
pip install -e .[dev, docs]
```

# Troubleshooting
If you have any issues installing, please [open an issue](https://github.com/cartercompbio/EUGENe/issues) on GitHub!