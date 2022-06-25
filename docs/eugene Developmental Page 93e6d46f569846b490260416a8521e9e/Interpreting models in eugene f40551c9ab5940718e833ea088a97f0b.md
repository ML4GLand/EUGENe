# Interpreting models in eugene

# Interpret

## Feature importances

- Deal with this issue with DeepLift: https://github.com/pytorch/pytorch/issues/57157
    - [https://captum.ai/docs/faq#can-my-model-using-functional-non-linearities-eg-nnfunctionalrelu-or-reused-modules-be-used-with-captum](https://captum.ai/docs/faq#can-my-model-using-functional-non-linearities-eg-nnfunctionalrelu-or-reused-modules-be-used-with-captum)

## PWMs for conv layers

In order to make models implemented using deepRAM easily interpretable, we extract motifs from the first convolutional layer following a similar methodology as in DeepBind (Alipanahi *et al.*, 2015). To do so, we feed all test sequences through the convolution stage. For each filter, we extract all sequence fragments that activate the filter and use only activations that are greater than half of the filter’s maximum value. Once all the sequence fragments are extracted, they are stacked and the nucleotide frequencies are counted to form a position frequency matrix (PFM). Sequence logos are then constructed using WebLogo (Crooks *et al.*, 2004). Finally, the discovered motifs are aligned using TOMTOM (Gupta *et al.*, 2007) against known motifs from CISBP-RNA (Ray *et al.*, 2013) for RBPs and JASPAR (Mathelier *et al.*, 2014) for transcription factors.

- How should we grab these from the model?
    - Have somewhat slow looping code “working” `/test/6B_test_EUGENE_interpretation-PWMs.ipynb`
        - I implemented the above with a combo of [PyTorch Lightning](https://www.notion.so/PyTorch-Lightning-8c6fd6cfa6de4011a50902302b6836dc) config code, [PyTorch](https://www.notion.so/PyTorch-09f8574b83144432aee712803ca1ea71)  code, looping and [seqlogo](https://www.notion.so/seqlogo-3c96a7adc8bc4b43816986b2b3e2dafa)
    - I would like to test this a little more rigorously

### Help

- [https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254](https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254)
- [https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3](https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3)
- [https://davetang.org/muse/2013/10/01/position-weight-matrix/](https://davetang.org/muse/2013/10/01/position-weight-matrix/)

## Other layer analyses

## In silico perturbations

- I need an *in silico* perturbation module maybe within the interpret
    - I see interpretations as
        1. conv filter analysis
        2. in silico mut
        3. feature importance via explanation methods

## `interpret.py`

## Within the `EUGENE` API