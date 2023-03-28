@track
def generate_pfms_sdata(
    model: nn.Module,
    sdata,
    method: str = "Alipanahi15",
    vocab: str = "DNA",
    num_filters: int = None,
    threshold: float = 0.5,
    num_seqlets: int = 100,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cpu",
    transform_kwargs: dict = {},
    key_name: str = "pfms",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
    **kwargs
):
    """
    Generate position frequency matrices for the maximal activating seqlets in the first 
    convolutional layer of a model. This involves computing the activations of the layer
    and then getting the sequences that activate the filter the most. We currently implement
    two methods, Alipanahi15 and Minnoye20. Using the maximally activating seqlets we then
    generate position frequency matrices for each filter and store those in the uns variable
    of the sdata object.

    Parameters
    ----------
    model
        The model to generate the PFMs with
    sdata
        The SeqData object holding sequences and to store the PFMs in
    method : str, optional
        The method to use, by default "Alipanahi15". This takes the all
        seqlets that activate the filter by more than half its maximum activation
    vocab : str, optional
        The vocabulary to use when decoding the sequences to create the PFM, by default "DNA"
    num_filters : int, optional
        The number of filters to get seqlets for, by default None. If not none will take the first
        num_filters filters in the model
    threshold : float, optional
        For Alipanahi15 method, the threshold defining maximally activating seqlets, by default 0.5
    num_seqlets : int, optional
        For Minnoye20 method, the number of seqlets to get, by default 100
    batch_size : int, optional
        The batch size to use when computing activations, by default None
    num_workers : int, optional
        The number of workers to use when computing activations, by default None
    device : str, optional
        The device to use when computing activations, by default "cpu" but will use gpu automatically if
        available
    transform_kwargs : dict, optional
        The kwargs to use when transforming the sequences when dataloading, by default ({})
        no arguments are passed (i.e. sequences are assumed to be ready for dataloading)
    key_name : str, optional
        The key to use when storing the PFMs in the uns variable of the sdata object, by default "pfms"
    prefix : str, optional
        The prefix to use when storing the PFMs in the uns variable of the sdata object, by default ""
    suffix : str, optional
        The suffix to use when storing the PFMs in the uns variable of the sdata object, by default ""
    copy : bool, optional
        Whether to return a copy the sdata object, by default False and sdata is modified in place
    """
    sdata = sdata.copy() if copy else sdata
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = DataLoader(
        sdataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    first_layer = _get_first_conv_layer(
        model, 
        device=device
    )
    activations, sequences = _get_activations_from_layer(
        first_layer, 
        sdataloader, 
        device=device, 
        vocab=vocab
    )
    filter_activators = _get_filter_activators(
        activations,
        sequences,
        first_layer.kernel_size[0],
        num_filters=num_filters,
        method=method,
        threshold=threshold,
        num_seqlets=num_seqlets,
    )
    filter_pfms = _get_pfms(
        filter_activators, 
        first_layer.kernel_size[0], 
        vocab=vocab
    )
    sdata.uns[f"{prefix}{key_name}{suffix}"] = filter_pfms
    return sdata if copy else None