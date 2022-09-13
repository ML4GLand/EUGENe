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
    model, 
    inputs, 
    ism_type="naive", 
    score_type="delta", 
    device=None, 
    batch_size=None
):
    batch_size = batch_size if batch_size is not None else settings.batch_size
    device = device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if ism_type == "naive":
        if model.strand != "ss":
            raise ValueError("Naive ISM currrently only works for single strand models")
        attrs = _naive_ism(model=model, X_0=inputs, type=score_type, device=device, batch_size=batch_size)
    else:
        raise ValueError("ISM type not supported")
    return attrs


def _grad_explain(
    model, 
    inputs, 
    target=None, 
    device="cpu"
):
    device = device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval()
    model.to(device)
    grad_explainer = InputXGradient(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    if model.strand == "ss":
        attrs = grad_explainer.attribute(
            forward_inputs, target=target, additional_forward_args=reverse_inputs
        )
        return attrs.to("cpu").detach().numpy()
    else:
        attrs = grad_explainer.attribute(
            (forward_inputs, reverse_inputs), target=target
        )
        return (attrs[0].to("cpu").detach().numpy(), attrs[1].to("cpu").detach().numpy())


def _deeplift_explain(
    model, 
    inputs, 
    ref_type="zero", 
    target=None, 
    device="cpu"
):
    if model.strand == "ds":
        raise ValueError("DeepLift currently only works for ss and ts strand models")
    device = device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
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
            torch.tensor([dinuc_shuffle_seq(forward_inputs.detach().to("cpu").numpy()[i].transpose()).transpose() for i in range(forward_inputs.shape[0])])
            .requires_grad_()
            .to(device)
        )
        reverse_ref = (
            torch.tensor([dinuc_shuffle_seq(reverse_inputs.detach().to("cpu").numpy()[i].transpose()).transpose() for i in range(reverse_inputs.shape[0])])
            .requires_grad_()
            .to(device)
        )
    elif ref_type == "gc":
        forward_ref = (
            torch.tensor([0.3, 0.2, 0.2, 0.3])
            .expand(forward_inputs.size()[2], 4)
            .unsqueeze(dim=0)
            .to(device).transpose(2,1)
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
            (forward_inputs, reverse_inputs), baselines=(forward_ref, reverse_ref), target=target
        )
        return (attrs[0].to("cpu").detach().numpy(), attrs[1].to("cpu").detach().numpy())


def _gradientshap_explain(model, inputs, ref_type="zero", target=None, device="cpu"):
    device = device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
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
            torch.tensor([dinuc_shuffle_seq(forward_inputs.detach().to("cpu").numpy()[i].transpose()).transpose() for i in range(forward_inputs.shape[0])])
            .requires_grad_()
            .to(device)
        )
        reverse_ref = (
            torch.tensor([dinuc_shuffle_seq(reverse_inputs.detach().to("cpu").numpy()[i].transpose()).transpose() for i in range(reverse_inputs.shape[0])])
            .requires_grad_()
            .to(device)
        )
    elif ref_type == "gc":
        forward_ref = (
            torch.tensor([0.3, 0.2, 0.2, 0.3])
            .expand(forward_inputs.size()[2], 4)
            .unsqueeze(dim=0)
            .to(device).transpose(2,1)
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
            (forward_inputs, reverse_inputs), baselines=(forward_ref, reverse_ref), target=target
        )
        return (attrs[0].to("cpu").detach().numpy(), attrs[1].to("cpu").detach().numpy())


def nn_explain(
    model,
    inputs,
    saliency_type,
    target=None,
    ref_type="zero",
    device="cpu",
    batch_size=None,
    abs_value=False,
    **kwargs
):
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
            **kwargs
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
    model,
    sdata,
    method="InputXGradient",
    target=None,
    batch_size: int = None,
    num_workers: int = None,
    device="cpu",
    transform_kwargs={},
    prefix="",
    suffix="",
    copy=False,
    **kwargs
):  
    torch.backends.cudnn.enabled = False
    print(torch.backends.cudnn.enabled)
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
        desc=f"Computing saliency on batches of size {batch_size}"
    ):
        _, x, x_rev_comp, y = batch
        curr_explanations = nn_explain(
            model,
            (x, x_rev_comp),
            target=target,
            saliency_type=method,
            device=device,
            batch_size=batch_size,
            **kwargs
        )
        if (i_batch + 1) * batch_size < dataset_len:
            if model.strand == "ss":
                all_forward_explanations[i_batch * batch_size : (i_batch + 1) * batch_size] = curr_explanations
            else:
                all_forward_explanations[i_batch * batch_size : (i_batch + 1) * batch_size] = curr_explanations[0]
                all_reverse_explanations[i_batch * batch_size : (i_batch + 1) * batch_size] = curr_explanations[1]
        else:
            if model.strand == "ss":
                all_forward_explanations[i_batch * batch_size : dataset_len] = curr_explanations
            else:
                all_forward_explanations[i_batch * batch_size : dataset_len] = curr_explanations[0]
                all_reverse_explanations[i_batch * batch_size : dataset_len] = curr_explanations[1]
    if model.strand == "ss":
        sdata.uns[f"{prefix}{method}_imps{suffix}"] = all_forward_explanations
    else:
        sdata.uns[f"{prefix}{method}_forward_imps{suffix}"] = all_forward_explanations
        sdata.uns[f"{prefix}{method}_reverse_imps{suffix}"] = all_reverse_explanations
    return sdata if copy else None


@track
def aggregate_importances_sdata(
    sdata, 
    uns_key
):
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
