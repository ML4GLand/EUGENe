import numpy as np
import torch
from captum.attr import InputXGradient, DeepLift
from ..utils import track
from ..preprocessing import dinuc_shuffle
from tqdm.auto import tqdm


def grad_explain(model, inputs, ref_type=None, target=None, device="cpu"):
    model.train()
    model.to(device)
    grad_explainer = InputXGradient(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    attrs = grad_explainer.attribute(forward_inputs, additional_forward_args=reverse_inputs)
    return attrs.to("cpu").detach().numpy()


def deeplift_explain(model, inputs, ref_type="zero", target=None, device="cpu"):
    model.train()
    model.to(device)
    deep_lift = DeepLift(model)
    forward_inputs = inputs[0].requires_grad_().to(device)
    reverse_inputs = inputs[1].requires_grad_().to(device)
    print(ref_type)
    if ref_type == "zero":
        forward_ref = torch.zeros(inputs[0].size()).to(device)
        reverse_ref = torch.zeros(inputs[1].size()).to(device)
    elif ref_type == "shuffle":
        forward_shuffled = forward_inputs.detach().to("cpu").squeeze(dim=0).numpy()
        forward_shuffled = reverse_inputs.detach().to("cpu").squeeze(dim=0).numpy()
        forward_ref = torch.tensor(dinuc_shuffle(forward_shuffled)).unsqueeze(dim=0).requires_grad_().to(device)
        reverse_ref = torch.tensor(dinuc_shuffle(reverse_ref)).unsqueeze(dim=0).requires_grad_().to(device)
    elif ref_type == "gc":
        ref = torch.tensor([0.3, 0.2, 0.2, 0.3]).expand(forward_inputs.size()[1], 4).unsqueeze(dim=0).to(device)
    attrs = deep_lift.attribute(inputs=forward_inputs, baselines=forward_ref, additional_forward_args=reverse_inputs)
    return attrs.to("cpu").detach().numpy()


def nn_explain(model,
               inputs,
               saliency_type,
               target=None,
               ref_type="zero",
               device="cpu",
               abs_value=False):
    if saliency_type == "DeepLift":
        attrs = deeplift_explain(model=model, inputs=inputs, ref_type=ref_type, device=device, target=target)
    elif saliency_type == "InputXGradient":
        attrs = grad_explain(model=model, inputs=inputs, ref_type=ref_type, device=device, target=target)
    if abs_value:
        attrs = np.abs(attrs)
    return attrs


@track
def feature_attribution(model, sdata, batch_size=32, saliency_method="InputXGradient", device="cpu", copy=False):

    sdata = sdata.copy() if copy else sdata
    sdataset = sdata.to_dataset(label=None, seq_transforms=["one_hot_encode"], transform_kwargs={"transpose": True})
    sdataloader = sdataset.to_dataloader(batch_size=batch_size)

    dataset_len = len(sdataloader.dataset)
    example_shape = sdataloader.dataset[0][1].numpy().shape
    all_explanations = np.zeros((dataset_len, *example_shape))
    if batch_size == None:
        batch_size = sdataloader.batch_size
    for i_batch, batch in tqdm(enumerate(sdataloader), total=int(dataset_len / batch_size)):
        ID, x, x_rev_comp, y = batch
        curr_explanations = nn_explain(model, (x, x_rev_comp), saliency_type=saliency_method, device=device)
        if (i_batch+1)*batch_size < dataset_len:
            #print(i_batch*BATCH_SIZE, (i_batch+1)*BATCH_SIZE)
            all_explanations[i_batch*batch_size: (i_batch+1)*batch_size] = curr_explanations
        else:
            #print(i_batch*BATCH_SIZE, dataset_len)
            all_explanations[i_batch*batch_size:dataset_len] = curr_explanations
    sdata.uns[f"{saliency_method}_imps"] = all_explanations
    return sdata if copy else None
