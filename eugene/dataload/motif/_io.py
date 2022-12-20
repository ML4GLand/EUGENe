from ._Motif import Motif, MotifSet
from ._utils import _parse_version, _parse_alphabet, _parse_strands, _parse_background, _parse_motif

def read_meme(
    filename
):
    motifs = {}
    version = None
    alphabet = None
    strands = None
    background = None
    background_source = None
    with open(filename) as meme_file:
        line = meme_file.readline()
        version = _parse_version(line)
        line = meme_file.readline()
        while line:
            if line.startswith("ALPHABET"):
                if alphabet is None:
                    alphabet = _parse_alphabet(line)
                    line = meme_file.readline()
                else:
                    raise RuntimeError("Multiple alphabet definitions encountered in MEME file")
            elif line.startswith("strands: "):
                if strands is None:
                    strands = _parse_strands(line)
                    line = meme_file.readline()
                else:
                    raise RuntimeError("Multiple strand definitions encountered in MEME file")
            elif line.startswith("Background letter frequencies"):
                if background is None:
                    line = _parse_background(line, meme_file)
                else:
                    raise RuntimeError("Multiple background frequency definitions encountered in MEME file")
            elif line.startswith("MOTIF"):
                motif = _parse_motif(line, meme_file)
                if motif.identifier in motifs:
                    raise RuntimeError("Motif identifiers not unique within file")
                motifs[motif.identifier] = motif
                line = meme_file.readline()
            else:
                line = meme_file.readline()
    return MotifSet(
        motifs=motifs,
        version=version,
        alphabet=alphabet,
        strands=strands,
        background=background,
        background_source=background_source,
    )


def read_jaspar(
    motif_accs=None,
    motif_names=None,
    collection=None,
    release="JASPAR2022",
    identifier_number=0,
    verbose=False,
    **kwargs,   
):
    """Read motifs from JASPAR database

    Parameters
    ----------
    motif_accs : list of str, optional
        List of motif accessions, by default None
    motif_names : list of str, optional
        List of motif names, by default None
    collection : str, optional
        Collection name, by default None
    release : str, optional
        JASPAR release, by default "JASPAR2020"
    identifier_number : int, optional
        Identifier number, by default 0
    verbose : bool, optional
        Verbose, by default False

    Returns
    -------
    MotifSet
    """
    motifs = _load_pyjaspar(
        motif_accs=motif_accs,
        motif_names=motif_names,
        collection=collection,
        release=release,
        **kwargs,
    )
    motif_set = _from_biopython_motifs(motifs, identifier_number=identifier_number, verbose=verbose)
    return motif_set

    

def write_matrix(
    pwm, 
    output_file_filename, 
    vocab="DNA"
):
    """
    Function to convert pwm as ndarray to meme file
    Adapted from:: nnexplain GitHub:

    Parameters
    ----------
    pwm:
        numpy.array, pwm matrices, shape (U, 4, filter_size), where U - number of units
    output_file_filename:
        string, the name of the output meme file
    """
    from ...preprocess._utils import _get_vocab

    vocab = "".join(_get_vocab(vocab))
    n_filters = pwm.shape[0]
    filter_size = pwm.shape[2]
    meme_file = open(output_file_filename, "w")
    meme_file.write("MEME version 4\n\n")
    meme_file.write(f"ALPHABET= {vocab}\n\n")
    meme_file.write("strands: + -\n\n")
    meme_file.write("Background letter frequencies\n")
    meme_file.write(
        f"{vocab[0]} 0.25 {vocab[1]} 0.25 {vocab[2]} 0.25 {vocab[3]} 0.25\n"
    )

    print("Saved PWM File as : {}".format(output_file_filename))

    for i in range(0, n_filters):
        if np.sum(pwm[i, :, :]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s\n" % i)
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n"
                % np.count_nonzero(np.sum(pwm[i, :, :], axis=0))
            )

        for j in range(0, filter_size):
            if np.sum(pwm[i, :, j]) > 0:
                meme_file.write(
                    str(pwm[i, 0, j])
                    + "\t"
                    + str(pwm[i, 1, j])
                    + "\t"
                    + str(pwm[i, 2, j])
                    + "\t"
                    + str(pwm[i, 3, j])
                    + "\n"
                )

    meme_file.close()

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
    output_file_filename:
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
        output_file_filename = os.filename.join(settings.output_dir, file_name)
    else:
        output_file_filename = os.filename.join(output_dir, file_name)
    pwms = np.array([pfms[key].values for key in filter_ids])
    if convert_to_pfm:
        pwms / pwms.sum(axis=2, keepdims=True)
    if change_length_axis:
        pwms = pwms.transpose(0, 2, 1)
    pwm_to_meme(pwms, output_file_filename, vocab=vocab)
    if return_pfms:
        return pwms