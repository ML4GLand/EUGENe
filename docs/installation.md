# Installation
EUGENe is a Python package, and can be installed using `pip`:

```bash
pip install 'eugene-tools'
```

Extras for development `[dev, docs]` can be installed using:

```bash
pip install 'eugene-tools[dev, docs]'
```

```python
import eugene as eu
```

```{note}
In it's current state, EUGENe is a pretty dependency heavy package. Some of this is due to the nature of the field, but we are also working on slimming down the package for future releases
```

## Bleeding edge version
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
If you want to edit the docs, you will need to run `pip install -e .[docs]
```

## Troubleshooting
If you have any issues installing, please [open an issue](https://github.com/cartercompbio/EUGENe/issues) on GitHub! We will do our best to help you ASAP!
