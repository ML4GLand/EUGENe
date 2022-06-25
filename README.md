# eugene

Cluster Path (if applicable): /cellar/users/aklie/projects/EUGENE
Concept(s): neural networks
Created: May 4, 2022 10:03 AM
Data type(s): MPRA
Developer: https://www.notion.so/0fe2c37754e54a75bf940150de1779b3 
Domain(s): sequence classification/regression
Environment(s): https://www.notion.so/seqtools-py37-9fb9e31c952e4b3d86a14bd324cabd66, https://www.notion.so/eugene_benchmarks-a6da09e2b46a4eb89d6e9ea710608510
Language (if applicable): Python
Last Edited Time: June 25, 2022 10:30 AM
Status: Developing
Type: model, tool
URL: https://github.com/adamklie/EUGENE

<aside>
👋 Welcome to the official `eugene` toolkit Notion page! This page is designed to be a one stop shop for developing and documenting the framework for biological sequence based predictive tasks.

</aside>

# ❓What is `eugene`?

EUGENE (**E**lucidating and **U**nderstanding **G**rammar of **E**nhancers with **Ne**uralnets) is a computational framework for building models that make predictions using primarily DNA sequence as input. EUGENE is primarily designed for predicting tissue-specific enhancer activity from massively parallel reporter assay (MPRA) libraries, but is general enough to be applied to a variety of sequence-based prediction tasks. It consists of a codebase for building, training, validating and interpreting several deep learning architectures. Some notes on the scope of **EUGENE:**

- **For starters, the models will only DNA input will be considered. But it is worth noting that extensions to other sequence based inputs are not out of the question**
- **Most of the tasks performed by this framework will be supervised (i.e., labels are necessary), though we are looking to extend into some more unsupervised, semi-supervised and probabilistic modeling techniques**
- 

# Installation

---

[seqtools-py37](https://www.notion.so/seqtools-py37-9fb9e31c952e4b3d86a14bd324cabd66) and [eugene_benchmarks](https://www.notion.so/eugene_benchmarks-a6da09e2b46a4eb89d6e9ea710608510) 

```bash
pip install -e eugene
```

Need a base set of packages necessary to get this up and running (eventually)

- `pytorch_lightning`(with CLI add-ons*)
- `pytorch`
- `seqlogo`
- 

# Core Functionality

---

1. Build a LightningModule
2. Train a LightningModule from the command line or through the API
3. Validate the model on unseen test data
4. Interpret the model through a suite of tools

---

# Development

[`eugene` Developmental Page](eugene%20ec67d8229638439e81349bdc48ff7476/eugene%20Developmental%20Page%2093e6d46f569846b490260416a8521e9e.md)

## Benchmarks

---

1. OHE sequence methods
2. 

## Potential collaborator and integration ideas

---

[deepRAM](https://www.notion.so/deepRAM-97fb05adf27b40878e7d68d5fd876665) 

[kipoi](https://www.notion.so/kipoi-f2ac6048f0e14ae0ad27aa6cb8f9e9a2) 

[Selene](https://www.notion.so/Selene-0cacf462544041f2af0766fb2f9f1132) 

[DeepLift](https://www.notion.so/DeepLift-1e2102bf3e8c45a4bfd30439e6f941ca) 

[ExplaiNN](https://www.notion.so/ExplaiNN-f022f066356e454a85105272791d0021) 

concise/ziga

stein methods

[GraphReg](https://www.notion.so/GraphReg-049a876f3bf44b319025985b695d9bb1) 

[scvi-tools](https://www.notion.so/scvi-tools-7e8e41d13e2b415485dcf75fd5dfff90): inspiration for structure

[TF-MoDisCo](https://www.notion.so/TF-MoDisCo-a08046f50fc64befaaaf567800c62123) 

## Help Wanted

---

I am actively recruiting to help me with this project. If you are interested in software development for building predictive models in sequence based genomics send me an