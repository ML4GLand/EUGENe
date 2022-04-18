import torch
from captum.attr import Saliency, DeepLift, GuidedGradCam

def nn_explain(model, 
               inputs, 
               saliency_type, 
               target=None, 
               ref_type=None,
               device="cpu",
               abs_value=False):
    if saliency_type == "DeepLift":
        attrs = deeplift_explain(model=model, inputs=inputs, ref_type=ref_type, device=device, target=target)
    elif saliency_type == "Saliency":
        attrs = vanilla_explain(model=model, inputs=inputs, ref_type=ref_type, device=device, target=target)
    elif saliency_type == "GuidedGradCam":
        attrs = gradcam_explain(model=model, inputs=inputs, ref_type=ref_type, device=device, target=target)
    if abs_value:
        attrs = np.abs(attrs)
    return attrs

def gradcam_explain(model, inputs, ref_type=None, target=None, device="cpu"):
    from captum.attr import GuidedGradCam
    model.train()
    model.to(device)
    vanilla = Saliency(model)
    inputs.requires_grad_()
    inputs = inputs.to(device)
    attrs = vanilla.attribute(inputs=inputs)
    return attrs.to("cpu").detach().numpy()

def vanilla_explain(model, inputs, ref_type=None, target=None, device="cpu"):
    from captum.attr import Saliency
    model.train()
    model.to(device)
    vanilla = Saliency(model)
    inputs.requires_grad_()
    inputs = inputs.to(device)
    attrs = vanilla.attribute(inputs=inputs)
    return attrs.to("cpu").detach().numpy()

def deeplift_explain(model, inputs, ref_type="zero", target=None, device="cpu"):
    import dinuc_shuffle
    from captum.attr import DeepLift
    model.train()
    model.to(device)
    deep_lift = DeepLift(model)
    inputs.requires_grad_()
    inputs = inputs.to(device)
    if ref_type == "zero":
        ref = torch.zeros(inputs.size()).to(device)
    elif ref_type == "shuffle":
        print(inputs.size())
        to_shuf = inputs.detach().to("cpu").squeeze(dim=0).numpy()
        ref = torch.tensor(dinuc_shuffle.dinuc_shuffle(to_shuf)).unsqueeze(dim=0).to(device)
    elif ref_type == "gc":
        ref = torch.tensor([0.3, 0.2, 0.2, 0.3]).expand(inp.size()[1], 4).unsqueeze(dim=0).to(device)
    ref.requires_grad_()
    attrs = deep_lift.attribute(inputs=inputs, baselines=ref)
    return attrs.to("cpu").detach().numpy()