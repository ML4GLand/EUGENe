# EUGENE (**E**lucidating and **U**nderstanding **G**rammar of **E**lements with **Ne**uralnets)

EUGENe represents a computational framework for building models of regulatory sequences as input. It is designed after the scverse framework for single cell analysis in Python and is meant to make the development in the deep learning genomics field more accessible. EUGENE consists of a codebase for building, training, validating and interpreting several deep learners that model sequence-based data. EUGENE is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. Jupyter), however we have also implemented several key functions via the command line.

---

# Developmental Installation (WIP)

```bash
conda create --name eugene_dev python=3.7 -y
conda activate eugene_dev
pip install -e .[docs]
pip install torch-tb-profiler pre-commit
conda install -c anaconda cudatoolkit=10.2
python -m ipykernel install --user --name eugene_dev --display-name "Python 3.7 eugene_dev"
```

---

# Core Functionality

**Load commonly used datasets or your own data**

**Prepare data for training a model**

**Build a LightningModule**

**Train a LightningModule from the Python API or via command line**

**Validate the model by making predictions on unseen test data**

**Interpret the model through a suite of methods**

**Visualize it all as you go**

**Benchmark against other common methods**

---

# Integrated Functionality
This package would not have been possible without the functionality of tools past:

- [`deepRAM`](https://www.notion.so/deepRAM-97fb05adf27b40878e7d68d5fd876665)
- [`kipoi`](https://www.notion.so/kipoi-f2ac6048f0e14ae0ad27aa6cb8f9e9a2)
- [`Selene`](https://www.notion.so/Selene-0cacf462544041f2af0766fb2f9f1132)
- [`DeepLift`](https://www.notion.so/DeepLift-1e2102bf3e8c45a4bfd30439e6f941ca)
- [`ExplaiNN`](https://www.notion.so/ExplaiNN-f022f066356e454a85105272791d0021)
- [`concise`](https://github.com/gagneurlab/concise/tree/master/concise)
- [`GraphReg`](https://www.notion.so/GraphReg-049a876f3bf44b319025985b695d9bb1)
- [`scvi-tools`](https://www.notion.so/scvi-tools-7e8e41d13e2b415485dcf75fd5dfff90)
- [`TF-MoDisCo`](https://www.notion.so/TF-MoDisCo-a08046f50fc64befaaaf567800c62123)

We have worked hard to abstract away as much of the technical details of these packages to allow the user a very smooth experience with their analyses. For power users looking to develop we recommend you check out more details of each package at the above links

---

# Help Wanted

I am [actively recruiting](https://www.notion.so/eugene-Team-Page-8d31da75046049fa86264be57e5711bf) contributors for this project! If you are interested in software development for building predictive models in sequence based genomics send me [an email](aklie@eng.ucsd.edu). All collaborators and developers will of course be included on all publications. See the end of this page for more info on expected skills and qualifications. Contact aklie@ucsd.eng.edu if you are interested in contributing to the project.

---
