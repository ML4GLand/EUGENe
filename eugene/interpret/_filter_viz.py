import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from ..preprocess._utils import _get_vocab
from ._utils import _k_largest_index_argsort
from ..utils import track
from .._settings import settings


def _get_first_conv_layer(
    model: nn.Module, 
    device: str = "cpu"
):
    """
    Get the first convolutional layer of a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the first convolutional layer from.
    device : str, optional
        The device to move the layer to, by default "cpu"

    Returns
    -------
    torch.nn.Module
        The first convolutional layer in the model.
    """
    if model.__class__.__name__ == "Jores21CNN":
        layer_shape = model.biconv.kernels[0].shape
        kernels = model.biconv.kernels[0]
        biases = model.biconv.biases[0]
        layer = nn.Conv1d(
            in_channels=layer_shape[1],
            out_channels=layer_shape[0],
            kernel_size=layer_shape[2],
            padding="same",
        )
        layer.weight = nn.Parameter(kernels)
        layer.bias = nn.Parameter(biases)
        return layer.to(device)
    elif model.__class__.__name__ == "Kopp21CNN":
        return model.conv
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            first_layer = model.convnet.module[0]
            return first_layer.to(device)
    print("No Conv1d layer found, returning None")
    return None


def _get_activations_from_layer(
    layer: nn.Module, 
    sdataloader: DataLoader,
    device: str = "cpu", 
    vocab: str = "DNA"
):
    """
    Get the values of activations using a passed in layer and sequence inputs in a dataloader
    TODO: We currently generate the sequences for all activators which involves decoding
    all of them. We only need to do this for the maximal activating seqlets

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to get activations from.
    sdataloader : DataLoader
        The dataloader to get sequences from.
    device : str, optional
        The device to move the layer to, by default "cpu"
    vocab : str, optional
        The vocabulary to use, by default "DNA"

    Returns
    -------
    np.ndarray
        The activations from the layer.

    Note
    ----
    We currently only use forward sequences for computing activations of the layer,
    we do not currenlty include reverse complements
    """
    from ..preprocess import decode_seqs
    activations = []
    sequences = []
    dataset_len = len(sdataloader.dataset)
    batch_size = sdataloader.batch_size
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc="Getting maximial activating seqlets",
    ):
        ID, x, x_rev_comp, y = batch
        sequences.append(decode_seqs(x.detach().cpu().numpy(), vocab=vocab, verbose=False))
        x = x.to(device)
        layer = layer.to(device)
        activations.append(F.relu(layer(x)).detach().cpu().numpy())
        np_act = np.concatenate(activations)
        np_seq = np.concatenate(sequences)
    return np_act, np_seq


def _get_filter_activators(
    activations: np.ndarray,
    sequences: np.ndarray,
    kernel_size: int,
    num_filters: int = None,
    method: str = "Alipanahi15",
    threshold: float = 0.5,
    num_seqlets: int = 100,
):
    """
    Get the sequences that activate a filter the most using a passed in method.
    We currently implement two methods, Alipanahi15 and Minnoye20.
    
    Parameters
    ----------
    activations : np.ndarray
        The activations from the layer.
    sequences : np.ndarray
        The sequences corresponding to the activations in a numpy array.
    kernel_size : int
        The kernel size of the layer.
    num_filters : int, optional
        The number of filters to get seqlets for, by default None
    method : str, optional
        The method to use, by default "Alipanahi15"
    threshold : float, optional
        The threshold for filtering activations, by default 0.5
    num_seqlets : int, optional
        The number of seqlets to get, by default 100

    Returns
    -------
    np.ndarray
        The sequences that activate the filter the most.

    Note
    ----
    We currently only use forward sequences for computing activations of the layer,
    we do not currenlty include reverse complements
    """
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    if method == "Alipanahi15":
        assert (threshold is not None), "Threshold must be specified for Alipanahi15 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            max_val = np.max(single_filter)
            activators = []
            for i in range(len(single_filter)):
                starts = np.where(single_filter[i] >= max_val * threshold)[0]
                for start in starts:
                    activators.append(sequences[i][start : start + kernel_size])
            filter_activators.append(activators)
    elif method == "Minnoye20":
        assert (num_seqlets is not None), "num_seqlets must be specified for Minnoye20 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            inds = _k_largest_index_argsort(single_filter, num_seqlets)
            filter_activators.append(
                [
                    seq[inds[i][1] : inds[i][1] + kernel_size]
                    for i, seq in enumerate(sequences[inds[:, 0]])
                ]
            )
    return filter_activators


def _get_pfms(
    filter_activators: np.ndarray,
    kernel_size: int,
    vocab: str = "DNA",
):  
    """
    Generate position frequency matrices for the maximal activating seqlets in filter_activators 

    Parameters
    ----------
    filter_activators : np.ndarray
        The sequences that activate the filter the most.
    kernel_size : int
        The kernel size of the layer.
    vocab : str, optional
        The vocabulary to use, by default "DNA"

    Returns
    -------
    np.ndarray
        The position frequency matrices for the maximal activating seqlets in filter_activators
    """
    filter_pfms = {}
    vocab = _get_vocab(vocab)
    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            vocab[0]: np.zeros(kernel_size),
            vocab[1]: np.zeros(kernel_size),
            vocab[2]: np.zeros(kernel_size),
            vocab[3]: np.zeros(kernel_size),
            "N": np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfm = filter_pfm.drop("N", axis=1)
        filter_pfms[i] = filter_pfm
        filter_pfms[i] = filter_pfms[i].div(filter_pfms[i].sum(axis=1), axis=0)
    return filter_pfms


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


def tomtom(motif_path, jaspar_path, output_path, evalue=False, thresh=0.5, dist='pearson', png=None, tomtom_path='tomtom'):
  """ perform tomtom analysis """
  "dist: allr |  ed |  kullback |  pearson |  sandelin"
  cmd = [tomtom_path,'-thresh', str(thresh), '-dist', dist]
  if evalue:
    cmd.append('-evalue')  
  if png:
    cmd.append('-png')
  cmd.extend(['-oc', output_path, motif_path, jaspar_path])

  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  return stdout, stderr

# possibly this too
#---------------------------------------------------------------------------------------
# evaluation of tomtom motif comparison

def match_hits_to_ground_truth(file_path, motifs, motif_names=None, num_filters=32):
  """ works with Tomtom version 5.1.0 
  inputs:
      - file_path: .tsv file output from tomtom analysis
      - motifs: list of list of JASPAR ids
      - motif_names: name of motifs in the list
      - num_filters: number of filters in conv layer (needed to normalize -- tomtom doesn't always give results for every filter)
  outputs:
      - match_fraction: fraction of hits to ground truth motifs
      - match_any: fraction of hits to any motif in JASPAR (except Gremb1)
      - filter_match: the motif of the best hit (to a ground truth motif)
      - filter_qvalue: the q-value of the best hit to a ground truth motif (1.0 means no hit)
      - motif_qvalue: for each ground truth motif, gives the best qvalue hit
      - motif_counts for each ground truth motif, gives number of filter hits
  """

  # add a zero for indexing no hits
  motifs = motifs.copy()
  motif_names = motif_names.copy()
  motifs.insert(0, [''])
  motif_names.insert(0, '')

  # get dataframe for tomtom results
  df = pd.read_csv(file_path, delimiter='\t')

  # loop through filters
  filter_qvalue = np.ones(num_filters)
  best_match = np.zeros(num_filters).astype(int)
  correction = 0  
  for name in np.unique(df['Query_ID'][:-3].to_numpy()):
    filter_index = int(name.split('r')[1])

    # get tomtom hits for filter
    subdf = df.loc[df['Query_ID'] == name]
    targets = subdf['Target_ID'].to_numpy()

    # loop through ground truth motifs
    for k, motif in enumerate(motifs): 

      # loop through variations of ground truth motif
      for id in motif: 

        # check if there is a match
        index = np.where((targets == id) ==  True)[0]
        if len(index) > 0:
          qvalue = subdf['q-value'].to_numpy()[index]

          # check to see if better motif hit, if so, update
          if filter_qvalue[filter_index] > qvalue:
            filter_qvalue[filter_index] = qvalue
            best_match[filter_index] = k 

    # dont' count hits to Gmeb1 (because too many)
    index = np.where((targets == 'MA0615.1') ==  True)[0]
    if len(index) > 0:
      if len(targets) == 1:
        correction += 1

  # get names of best match motifs
  filter_match = [motif_names[i] for i in best_match]

  # get hits to any motif
  num_matches = len(np.unique(df['Query_ID'])) - 3.  # 3 is correction because of last 3 lines of comments in the tsv file (may change across tomtom versions)
  match_any = (num_matches - correction)/num_filters  # counts hits to any motif (not including Grembl)

  # match fraction to ground truth motifs
  match_index = np.where(filter_qvalue != 1.)[0]
  if any(match_index):
    match_fraction = len(match_index)/float(num_filters)
  else:
    match_fraction = 0.  

  # get the number of hits and minimum q-value for each motif
  num_motifs = len(motifs) - 1
  motif_qvalue = np.zeros(num_motifs)
  motif_counts = np.zeros(num_motifs)
  for i in range(num_motifs):
    index = np.where(best_match == i+1)[0]
    if len(index) > 0:
      motif_qvalue[i] = np.min(filter_qvalue[index])
      motif_counts[i] = len(index)

  return match_fraction, match_any, filter_match, filter_qvalue, motif_qvalue, motif_counts


  def filters_to_meme_sdata(
    sdata,
    output_dir: str = None,
    file_name="filter.meme",
    uns_key="pfms",
    filter_ids: int = None,
    vocab="DNA",
    convert_to_pfm: bool = False,
    change_length_axis=True,
    return_pfms=False,
):
    """
    Function to convert a single filter to a meme file

    sdata:
        SingleData, single cell data
    filter_ids:
        int, index of the filter to convert
    outfile:
        string, the name of the output meme file
    convert_to_pwm:
        bool, whether to convert the filter to a pwm
    """
    try:
        pfms = sdata.uns.get(uns_key)
    except KeyError:
        print("No filters found in sdata.uns['{}']".format(uns_key))
    if filter_ids is None:
        filter_ids = list(sdata.uns[uns_key].keys())
    if output_dir is None:
        outfile = os.filename.join(settings.output_dir, file_name)
    else:
        outfile = os.filename.join(output_dir, file_name)
    pwms = np.array([pfms[key].values for key in filter_ids])
    if convert_to_pfm:
        pwms / pwms.sum(axis=2, keepdims=True)
    if change_length_axis:
        pwms = pwms.transpose(0, 2, 1)
    pwm_to_meme(pwms, outfile, vocab=vocab)
    if return_pfms:
        return pwms