import torch

def feature_attribution_sdata(
    model: torch.nn.Module,  # need to enforce this is a SequenceModel
    sdata,
    method: str = "DeepLiftShap",
    target: int = 0,
    aggr: str = None,
    multiply_by_inputs: bool = True,
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
    Wrapper function to compute feature attribution scores for a SequenceModel using the 
    set of sequences defined in a SeqData object.
    
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

    # Disable cudnn for faster computations
    torch.backends.cudnn.enabled = False
    
    # Copy the SeqData object if necessary
    sdata = sdata.copy() if copy else sdata

    # Configure the device, batch size, and number of workers
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers

    # Make a dataloader from the sdata
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = sdataset.to_dataloader(batch_size=batch_size, shuffle=False)
    
    # Create an empty array to hold attributions
    dataset_len = len(sdataloader.dataset)
    example_shape = sdataloader.dataset[0][1].numpy().shape
    all_forward_explanations = np.zeros((dataset_len, *example_shape))
    if model.strand != "ss":
        all_reverse_explanations = np.zeros((dataset_len, *example_shape))

    # Loop through batches and compute attributions
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc=f"Computing saliency on batches of size {batch_size}",
    ):
        _, x, x_rev_comp, y = batch
        if model.strand == "ss":
            curr_explanations = attribute(
                model,
                x,
                target=target,
                method=method,
                device=device,
                additional_forward_args=x_rev_comp[0],
                **kwargs,
            )
        else:
            curr_explanations = attribute(
                model,
                (x, x_rev_comp),
                target=target,
                method=method,
                device=device,
                **kwargs,
            )
        if (i_batch+1)*batch_size < dataset_len:
            if model.strand == "ss":
                all_forward_explanations[i_batch*batch_size:(i_batch+1)*batch_size] = curr_explanations.detach().cpu().numpy()
            else:
                all_forward_explanations[i_batch*batch_size:(i_batch+1)*batch_size] = curr_explanations[0].detach().cpu().numpy()
                all_reverse_explanations[i_batch * batch_size:(i_batch+1)*batch_size] = curr_explanations[1].detach().cpu().numpy()
        else:
            if model.strand == "ss":
                all_forward_explanations[i_batch * batch_size:dataset_len] = curr_explanations.detach().cpu().numpy()
            else:
                all_forward_explanations[i_batch*batch_size:dataset_len] = curr_explanations[0].detach().cpu().numpy()
                all_reverse_explanations[i_batch*batch_size:dataset_len] = curr_explanations[1].detach().cpu().numpy()
    
    # Add the attributions to sdata 
    if model.strand == "ss":
        sdata.uns[f"{prefix}{method}_imps{suffix}"] = all_forward_explanations
    else:
        if aggr == "max":
            sdata.uns[f"{prefix}{method}_imps{suffix}"] = np.maximum(all_forward_explanations, all_reverse_explanations)
        elif aggr == "mean":
            sdata.uns[f"{prefix}{method}_imps{suffix}"] = (all_forward_explanations + all_reverse_explanations) / 2
        elif aggr == None:
            sdata.uns[f"{prefix}{method}_forward_imps{suffix}"] = all_forward_explanations
            sdata.uns[f"{prefix}{method}_reverse_imps{suffix}"] = all_reverse_explanations
    return sdata if copy else None

def aggregate_importances_sdata(
    sdata, 
    uns_key,
    copy=False
):
    """
    Aggregate feature attribution scores for a SeqData
    
    This function aggregates the feature attribution scores for a SeqData object
    Parameters
    ----------
    sdata : SeqData
        SeqData object
    uns_key : str
        Key in the uns attribute of the SeqData object to use as feature attribution scores
    """
    sdata = sdata.copy() if copy else sdata
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
    return sdata if copy else None
