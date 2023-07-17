# Base installation
EUGENe is a Python package, and can be installed using `pip`:

```bash
pip install 'eugene-tools'
```

## Bleeding edge version (TODO)
To work with the latest version [on GitHub](https://github.com/cartercompbio/EUGENe), clone the repository and `cd` into its root directory.

```bash
git clone https://github.com/cartercompbio/EUGENe.git
cd EUGENe
```

Then, install the package in development mode:

```bash
pip install -e .[dev]
```

# Developmental installation
Extras for development `[dev, docs]` can be installed using:

```bash
pip install 'eugene-tools[dev, docs]'
```

Or if you are installing from source:

```bash
pip install -e .[docs]
```

## Troubleshooting
If you have any issues installing, please [open an issue](https://github.com/cartercompbio/EUGENe/issues) on GitHub! We will do our best to help you ASAP!
