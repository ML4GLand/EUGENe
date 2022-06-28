# EUGENE (*E*lucidating and *U*nderstanding *G*rammar of *E*nhancers with *Ne*uralnets)

EUGENE represents a computational framework for building models of regulatory sequences as input. It is designed after the scverse framework for single cell analysis in Python and is meant to make the developments in the deep learning genomics field more accessible. EUGENE consists of a codebase for building, training, validating and interpreting several deep learners that model sequence-based data. EUGENE is primarily designed to be used through its Python API and we feel that users will get the most out of it by using a notebook interface (i.e. Jupyter), however we have also implemented several key functions via the command line.

---

# Developmental Installation

```bash
conda create -f eugene_env.yml
pip install -e eugene
```

---

# Core Functionality

### **Load commonly used datasets and your own data**

### **Prepare data for training a model**

### **Build a LightningModule**

### **Train a LightningModule from the command line or through the API**

### **Validate the model on unseen test data**

### **Interpret the model through a suite of tools**

### **Visualize it all**

### **Benchmark against other common methods**

The core functionality of EUGENE is built off of several established packages in the neural nets and gene regulation fields. These include but are not limited to:

- `pytorch_lightning`(with CLI add-ons*)
- `pytorch`
- `seqlogo`

# Integrated Functionality
This package would not have been possible without the functionality of tools past:

- [`deepRAM`](https://www.notion.so/deepRAM-97fb05adf27b40878e7d68d5fd876665)
- [`kipoi](https://www.notion.so/kipoi-f2ac6048f0e14ae0ad27aa6cb8f9e9a2)
- [`Selene`](https://www.notion.so/Selene-0cacf462544041f2af0766fb2f9f1132)
- [`DeepLift`](https://www.notion.so/DeepLift-1e2102bf3e8c45a4bfd30439e6f941ca)
- [`ExplaiNN`](https://www.notion.so/ExplaiNN-f022f066356e454a85105272791d0021)
- `concise`/ziga
- stein methods
- [GraphReg](https://www.notion.so/GraphReg-049a876f3bf44b319025985b695d9bb1)
- [scvi-tools](https://www.notion.so/scvi-tools-7e8e41d13e2b415485dcf75fd5dfff90): inspiration for structure
- [TF-MoDisCo](https://www.notion.so/TF-MoDisCo-a08046f50fc64befaaaf567800c62123)

We have worked hard to abstract away as much of the technical details of these packages to allow the user a very smooth experience with their analyses. For power users looking to develop we recommend you check out more details of each package at the above links

---

## Help Wanted
eugene Team Page

---

I am actively recruiting to help me with this project. If you are interested in software development for building predictive models in sequence based genomics send me an
