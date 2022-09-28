import numpy as np
import torch
from tqdm.auto import tqdm
import pyranges as pr
from captum.attr import InputXGradient, DeepLift, GradientShap
from yuzu.naive_ism import naive_ism
from ..preprocess import dinuc_shuffle_seq, perturb_seqs
from ..utils import track
from ._utils import _naive_ism
from .._settings import settings


def _ism_explain(
    model: torch.nn.Module, 
    inputs: tuple, 
    ism_type: str = "naive", 
    score_type: str = "delta", 
    device: str = None, 
    batch_size: int = None
):
    """
    Compute in silico saturation mutagenesis scores using a model on a set of inputs

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to use for computing ISM scores. Can be a EUGENe trained model or one you trained with PyTorch
        or PL
    inputs : tuple
        Tuple of forward and reverse inputs to compute ISM scores on. If the model is a ss model, then the tuple
        should only contain the forward inputs
    ism_type: str
        Type of ISM to use. We currently only support naive ISM
    score_type: str
        Type of score to compute, currently support "delta", "l1" or "l2"
    device: str
        Device to use for computing ISM scores. If None, will use the device specified from the EUGENe settings
    batch_size: int
        Batch size to use for computing ISM scores. If None, will use the batch size specified from the EUGENe settings
    
    Returns
    -------
    nd.array
        Array of ISM scores
    """
    batch_size = batch_size if batch_size is not None else settings.batch_size
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    model.to(device)
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if ism_type == "naive":
        if model.strand != "ss":
            raise ValueError("Naive ISM currrently only works for single strand models")
        attrs = _naive_ism(
            model=model,
            X_0=inputs,
            type=score_type,
            device=device,
            batch_size=batch_size,
        )
    else:
        raise ValueError("ISM type not supported")
    return attrs


def _grad_explain(
    model: torch.nn.Module, 
    inputs: tuple,
    target: int = None, 
    device: str = "cpu"
):
    """
    Compute InputXGradient feature attribution scores using a model on a set of inputs.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to use for computing InputXGradient scores. 
        Can be a EUGENe trained model or one you trained with PyTorch or PL.
    inputs : tuple
        Tuple of forward and reverse complement inputs to compute InputXGradient scores on. 
        If the model is a ss model, then the scores will only be computed on the forward inputs.
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    device: str
        Device to use for computing InputXGradient scores. 
        EUGENe will always use a gpu if available
    """
    device "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    model.to(device)
    grad_explainer = InputXGradient(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    if model.strand == "ss":
        attrs = grad_explainer.attribute(
            forward_inputs, 
            target=target, 
            additional_forward_args=reverse_inputs
        )
        return attrs.to("cpu").detach().numpy()
    else:
        attrs = grad_explainer.attribute(
            (forward_inputs, reverse_inputs), 
            target=target
        )
        return (
            attrs[0].to("cpu").detach().numpy(),
            attrs[1].to("cpu").detach().numpy(),
        )


def _deeplift_explain(
    model: torch.nn.Module, 
    inputs: tuple,
    ref_type: str = "zero", 
    target: int = None, 
    device: str = "cpu"
):
    """
    Compute DeepLIFT feature attribution scores using a model on a set of inputs.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to use for computing DeepLIFT scores.
        Can be a EUGENe trained model or one you trained with PyTorch or PL.
    inputs : tuple
        Tuple of forward and reverse complement inputs to compute DeepLIFT scores on.
        If the model is a ss model, then the scores will only be computed on the forward inputs.
    ref_type: str
        Type of reference to use for computing DeepLIFT scores. By default this is an all zeros reference,
        but we also support a dinucleotide shuffled reference and one based on GC content
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    device: str
        Device to use for computing DeepLIFT scores.
        EUGENe will always use a gpu if available
    
    Returns
    -------
    nd.array
        Array of DeepLIFT scores
    """
    if model.strand == "ds":
        raise ValueError("DeepLift currently only works for ss and ts strand models")
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    model.to(device)
    deeplift_explainer = DeepLift(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    if ref_type == "zero":
        forward_ref = torch.zeros(inputs[0].size()).to(device)
        reverse_ref = torch.zeros(inputs[1].size()).to(device)
    elif ref_type == "shuffle":
        from ..preprocess import dinuc_shuffle_seq
        forward_ref = (
            torch.tensor(
                [
                    dinuc_shuffle_seq(
                        forward_inputs.detach().to("cpu").numpy()[i].transpose()
                    ).transpose()
                    for i in range(forward_inputs.shape[0])
                ]
            )
            .requires_grad_()
            .to(device)
        )
        reverse_ref = (
            torch.tensor(
                [
                    dinuc_shuffle_seq(
                        reverse_inputs.detach().to("cpu").numpy()[i].transpose()
                    ).transpose()
                    for i in range(reverse_inputs.shape[0])
                ]
            )
            .requires_grad_()
            .to(device)
        )
    elif ref_type == "gc":
        forward_ref = (
            torch.tensor([0.3, 0.2, 0.2, 0.3])
            .expand(forward_inputs.size()[2], 4)
            .unsqueeze(dim=0)
            .to(device)
            .transpose(2, 1)
        )
        reverse_ref = forward_ref.clone()
    if model.strand == "ss":
        attrs = deeplift_explainer.attribute(
            forward_inputs,
            baselines=forward_ref,
            target=target,
            additional_forward_args=reverse_inputs,
        )
        return attrs.to("cpu").detach().numpy()
    else:
        attrs = deeplift_explainer.attribute(
            (forward_inputs, reverse_inputs),
            baselines=(forward_ref, reverse_ref),
            target=target,
        )
        return (
            attrs[0].to("cpu").detach().numpy(),
            attrs[1].to("cpu").detach().numpy(),
        )


def _gradientshap_explain(
    model: torch.nn.Module,
    inputs: tuple,
    ref_type: str = "zero", 
    target: int = None,
    device: str = "cpu"
):
    """
    Compute GradientSHAP feature attribution scores using a model on a set of inputs.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to use for computing GradientSHAP scores.
    Can be a EUGENe trained model or one you trained with PyTorch or PL.
    inputs : tuple
        Tuple of forward and reverse complement inputs to compute GradientSHAP scores on.
        If the model is a ss model, then the scores will only be computed on the forward inputs.
    ref_type: str
        Type of reference to use for computing GradientSHAP scores. By default this is an all zeros reference,
        but we also support a dinucleotide shuffled reference and one based on GC content
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    device: str
        Device to use for computing GradientSHAP scores.
        EUGENe will always use a gpu if available
    
    Returns
    -------
    nd.array
        Array of GradientSHAP scores
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    model.to(device)
    gradientshap_explainer = GradientShap(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    if ref_type == "zero":
        forward_ref = torch.zeros(inputs[0].size()).to(device)
        reverse_ref = torch.zeros(inputs[1].size()).to(device)
    elif ref_type == "shuffle":
        from ..preprocess import dinuc_shuffle_seq

        forward_ref = (
            torch.tensor(
                [
                    dinuc_shuffle_seq(
                        forward_inputs.detach().to("cpu").numpy()[i].transpose()
                    ).transpose()
                    for i in range(forward_inputs.shape[0])
                ]
            )
            .requires_grad_()
            .to(device)
        )
        reverse_ref = (
            torch.tensor(
                [
                    dinuc_shuffle_seq(
                        reverse_inputs.detach().to("cpu").numpy()[i].transpose()
                    ).transpose()
                    for i in range(reverse_inputs.shape[0])
                ]
            )
            .requires_grad_()
            .to(device)
        )
    elif ref_type == "gc":
        forward_ref = (
            torch.tensor([0.3, 0.2, 0.2, 0.3])
            .expand(forward_inputs.size()[2], 4)
            .unsqueeze(dim=0)
            .to(device)
            .transpose(2, 1)
        )
        reverse_ref = forward_ref.clone()
    if model.strand == "ss":
        attrs = gradientshap_explainer.attribute(
            forward_inputs,
            baselines=forward_ref,
            target=target,
            additional_forward_args=reverse_inputs,
        )
        return attrs.to("cpu").detach().numpy()
    else:
        attrs = gradientshap_explainer.attribute(
            (forward_inputs, reverse_inputs),
            baselines=(forward_ref, reverse_ref),
            target=target,
        )
        return (
            attrs[0].to("cpu").detach().numpy(),
            attrs[1].to("cpu").detach().numpy(),
        )


def nn_explain(
    model: torch.nn.Module,
    inputs: tuple,  
    saliency_type: str,
    target: int = None,
    ref_type: str = "zero",
    device: str = "cpu",
    batch_size: int = None,
    abs_value: bool = False,
    **kwargs
):
    """
    Wrapper function for computing feature attribution scores using a model on a set of inputs.
    Allows for computing scores using different methods and different reference types on any task.

    Parameters
    ----------
    model : torch.nn.Module
       PyTorch model to use for computing feature attribution scores.
        Can be a EUGENe trained model or one you trained with PyTorch or PL.
    inputs : tuple
        Tuple of forward and reverse complement inputs to compute feature attribution scores on.
        If the model is a ss model, then the scores will only be computed on the forward inputs.
    saliency_type: str
        Type of saliency to use for computing feature attribution scores.
        Can be one of the following:
        - "saliency" (vanilla saliency)
        - "grad" (vanilla gradients)
        - "gradxinput" (gradients x inputs)
        - "intgrad" (integrated gradients)
        - "intgradxinput" (integrated gradients x inputs)
        - "smoothgrad" (smooth gradients)
        - "smoothgradxinput" (smooth gradients x inputs)
        - "deeplift" (DeepLIFT)
        - "gradientshap" (GradientSHAP)
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    ref_type: str
        Type of reference to use for computing feature attribution scores. By default this is an all zeros reference,
        but we also support a dinucleotide shuffled reference and one based on GC content
    device: str
        Device to use for computing feature attribution scores.
        EUGENe will always use a gpu if available
    batch_size: int
        Batch size to use for computing feature attribution scores. If not specified, will use the
        default batch size of the model
    abs_value: bool
        Whether to take the absolute value of the scores. By default this is False
    **kwargs
        Additional arguments to pass to the saliency method. For example, you can pass the number of
        samples to use for SmoothGrad and Integrated Gradients

    Returns
    -------
    nd.array
        Array of feature attribution scores 
    """
    model.eval()
    if saliency_type == "DeepLift":
        attrs = _deeplift_explain(
            model=model, inputs=inputs, ref_type=ref_type, device=device, target=target
        )
    elif saliency_type == "InputXGradient":
        attrs = _grad_explain(model=model, inputs=inputs, device=device, target=target)
    elif saliency_type == "NaiveISM":
        attrs = _ism_explain(
            model=model,
            inputs=inputs,
            ism_type="naive",
            device=device,
            batch_size=batch_size,
            **kwargs,
        )
    elif saliency_type == "GradientSHAP":
        attrs = _gradientshap_explain(
            model=model, inputs=inputs, ref_type=ref_type, device=device, target=target
        )
    else:
        raise ValueError("Saliency type not supported")
    if abs_value:
        attrs = np.abs(attrs)
    return attrs


@track
def feature_attribution_sdata(
    model: torch.nn.Module,
    sdata,
    method: str = "InputXGradient",
    target: int = None,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cpu",
    transform_kwargs: dict = {},
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
    **kwargs
):
    """
    Wrapper function to compute feature attribution scores for a set of sequences in a SeqData object.
    Allows for computing scores using different methods and different reference types on any task.

    Parameters
    ----------
    model : torch.nn.Module
       PyTorch model to use for computing feature attribution scores.
        Can be a EUGENe trained model or one you trained with PyTorch or PL.
    sdata : SeqData
        SeqData object containing the sequences to compute feature attribution scores on.
    method: str
        Type of saliency to use for computing feature attribution scores.
        Can be one of the following:
        - "saliency" (vanilla saliency)
        - "grad" (vanilla gradients)
        - "gradxinput" (gradients x inputs)
        - "intgrad" (integrated gradients)
        - "intgradxinput" (integrated gradients x inputs)
        - "smoothgrad" (smooth gradients)
        - "smoothgradxinput" (smooth gradients x inputs)
        - "deeplift" (DeepLIFT)
        - "gradientshap" (GradientSHAP)
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    batch_size: int
        Batch size to use for computing feature attribution scores. If not specified, will use the
        default batch size of the model
    num_workers: int
        Number of workers to use for computing feature attribution scores. If not specified, will use
        the default number of workers of the model
    device: str
        Device to use for computing feature attribution scores.
        EUGENe will always use a gpu if available
    transform_kwargs: dict
        Dictionary of keyword arguments to pass to the transform method of the model
    prefix: str
        Prefix to add to the feature attribution scores
    suffix: str
        Suffix to add to the feature attribution scores
    copy: bool
        Whether to copy the SeqData object before computing feature attribution scores. By default
        this is False
    **kwargs
        Additional arguments to pass to the saliency method. For example, you can pass the number of
        samples to use for SmoothGrad and Integrated Gradients

    Returns
    -------
    SeqData
        SeqData object containing the feature attribution scores
    """
    torch.backends.cudnn.enabled = False
    sdata = sdata.copy() if copy else sdata
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = sdataset.to_dataloader(batch_size=batch_size, shuffle=False)
    dataset_len = len(sdataloader.dataset)
    example_shape = sdataloader.dataset[0][1].numpy().shape
    all_forward_explanations = np.zeros((dataset_len, *example_shape))
    if model.strand != "ss":
        all_reverse_explanations = np.zeros((dataset_len, *example_shape))
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc=f"Computing saliency on batches of size {batch_size}",
    ):
        _, x, x_rev_comp, y = batch
        curr_explanations = nn_explain(
            model,
            (x, x_rev_comp),
            target=target,
            saliency_type=method,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )
        if (i_batch + 1) * batch_size < dataset_len:
            if model.strand == "ss":
                all_forward_explanations[
                    i_batch * batch_size : (i_batch + 1) * batch_size
                ] = curr_explanations
            else:
                all_forward_explanations[
                    i_batch * batch_size : (i_batch + 1) * batch_size
                ] = curr_explanations[0]
                all_reverse_explanations[
                    i_batch * batch_size : (i_batch + 1) * batch_size
                ] = curr_explanations[1]
        else:
            if model.strand == "ss":
                all_forward_explanations[
                    i_batch * batch_size : dataset_len
                ] = curr_explanations
            else:
                all_forward_explanations[
                    i_batch * batch_size : dataset_len
                ] = curr_explanations[0]
                all_reverse_explanations[
                    i_batch * batch_size : dataset_len
                ] = curr_explanations[1]
    if model.strand == "ss":
        sdata.uns[f"{prefix}{method}_imps{suffix}"] = all_forward_explanations
    else:
        sdata.uns[f"{prefix}{method}_forward_imps{suffix}"] = all_forward_explanations
        sdata.uns[f"{prefix}{method}_reverse_imps{suffix}"] = all_reverse_explanations
    return sdata if copy else None


@track
def aggregate_importances_sdata(sdata, uns_key):
    vals = sdata.uns[uns_key]
    df = sdata.pos_annot.df
    agg_scores = []
    for i, row in df.iterrows():
        seq_id = row["Chromosome"]
        start = row["Start"]
        end = row["End"]
        seq_idx = np.where(sdata.names == seq_id)[0][0]
        agg_scores.append(vals[seq_idx, :, start:end].sum())
    df[f"{uns_key}_agg_scores"] = agg_scores
    ranges = pr.PyRanges(df)
    sdata.pos_annot = ranges
