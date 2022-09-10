import os
import re
import pickle
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import logging
from vizsequence import viz_sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.preprocessing import OneHotEncoder
from ._seq_preprocess import ohe_seq
from ._utils import _merge_intervals, _hamming_distance, _collapse_pos


# Define these for use in any other function
file_abs_path = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(file_abs_path, '..', 'datasets/auxiliary')
ets_aff_file=f"{database_path}/parsed_Ets1_8mers.txt"
gata_aff_file=f"{database_path}/parsed_Gata6_3769_contig8mers.txt"
enhancer_binding_sites = {"Core-otx-a": ["..GGAA..", "..GGAT..", "..TTCC..", "..ATCC..", "..GATA..", "..TATC.."],
                          "WT-otx-a": ["GTTATCTC", "ACGGAAGT", "AAGGAAAT", "AATATCT", "AAGATAGG", "GAGATAAC", "ACTTCCGT", "ATTTCCTT", "AGATATT", "CCTATCTT"]}
alphabet = np.array(["A", "G", "C", "T"])

# Load Ets1 affinities into a dictionary with keys being all possible 8-mers and values being binding affinities (consensus=1)
def loadEtsAff(file):
    ref = file
    Seq2EtsAff  = {line.split('\t')[0]:float(line.split('\t')[1]) for line in open(ref,'r').readlines()}
    return Seq2EtsAff


# Load Gata6 Badis 2009 affinities into a dictionary with keys being all possible 8-mers and values being binding affinities (consensus=1)
def loadGata6Aff(file):
    ref = file
    Seq2GataAff = {line.split('\t')[0]:float(line.split('\t')[1]) for line in open(ref,'r').readlines()}
    return Seq2GataAff


# Load the affinities
ets_aff = loadEtsAff(ets_aff_file)
gata_aff = loadGata6Aff(gata_aff_file)


# Define for use in next two functions
BindingSiteName2affinities_file=f"{database_path}/bindingSiteName2affinities.pkl"
SiteName2bindingSiteSequence_file=f"{database_path}/siteName2bindingSiteSequence.pkl"


# Load Otx-a binding site to affinity dictionary
def loadBindingSiteName2affinities(file=BindingSiteName2affinities_file, pickle_obj=True):
    if pickle_obj:
        with open(file, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        print("Only pickles at this time")


# Load Otx-a binding site name to sequence dictionary
def loadSiteName2bindingSiteSequence(file=SiteName2bindingSiteSequence_file, pickle_obj=True):
    if pickle_obj:
        with open(file, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        print("Only pickles at this time")


### Sequence annotation ###
def randomizeLinkers(seq, features=None, enhancer=None):
    if features == None:
        assert enhancer != None
        features = enhancer_binding_sites[enhancer]

    transformed_seq = []
    feature_spans = _merge_intervals([x.span() for x in re.finditer(r"("+'|'.join(features)+r")", seq)])
    if feature_spans is None:
        return seq
    for i, span in enumerate(feature_spans):
        if i == 0:
            linker_len = span[0]
        else:
            linker_len = feature_spans[i][0]-feature_spans[i-1][1]
        transformed_seq.append("".join(np.random.choice(alphabet, size=linker_len)))
        transformed_seq.append(seq[span[0]:span[1]])
    transformed_seq.append("".join(np.random.choice(alphabet, size=len(seq)-feature_spans[-1][1])))
    transformed_seq = "".join(transformed_seq)
    if len(transformed_seq) != len(seq):
        logging.warning('Transformed sequence is length {}'.format(len(transformed_seq)))
    return transformed_seq


# Fit to overall dataframe
def encodeBlock(dataset, block_features):
    ohe_block = OneHotEncoder(sparse=False)
    X = dataset[block_features]
    ohe_block.fit(X)
    X_block = ohe_block.fit_transform(X)


# Function to return a dictionary with the position of the first nucleotide of every GATA and ETS core fond in an input sequence. Also includes orientation
def findEtsAndGataCores(seq, cores={"ETS_FORWARD": ["GGAA", "GGAT"], "ETS_REVERSE": ["TTCC", "ATCC"], "GATA_FORWARD": ["GATA"], "GATA_REVERSE": ["TATC"]}):
    core_pos = {}
    for i in range(2, len(seq)-5):
        if seq[i:i+4] in cores["ETS_FORWARD"]:
            core_pos.setdefault(i, []).append("ETS")
            core_pos[i].append("F")

        elif seq[i:i+4] in cores["ETS_REVERSE"]:
            core_pos.setdefault(i, []).append("ETS")
            core_pos[i].append("R")

        elif seq[i:i+4] in cores["GATA_FORWARD"]:
            core_pos.setdefault(i, []).append("GATA")
            core_pos[i].append("F")

        elif seq[i:i+4] in cores["GATA_REVERSE"]:
            core_pos.setdefault(i, []).append("GATA")
            core_pos[i].append("R")
    return core_pos


# Function to add the affinity and sequence of the binding site cores identified by findEtsAndGataCores()
def findTFBSAffinity(seq, cores, ets_aff_file="../datasets/parsed_Ets1_8mers.txt", gata_aff_file="../datasets/parsed_Gata6_3769_contig8mers.txt"):
    #ets_aff = loadEtsAff(ets_aff_file)
    #gata_aff = loadGata6Aff(gata_aff_file)
    for pos in cores.keys():
        cores[pos].append(seq[pos-2:pos+6])
        if cores[pos][0] == "ETS":
            cores[pos].append(ets_aff[seq[pos-2:pos+6]])
        elif cores[pos][0] == "GATA":
            cores[pos].append(gata_aff[seq[pos-2:pos+6]])
    return cores


# Function to add the spacing between binding sites given a core dictionary. Specifically adds the distance from the start of each binding site to the last binding site
def findSpacingBetweenTFBS(cores):
    sorted_core_pos = sorted(list(cores.keys()))
    previous_pos = 0
    for i, pos in enumerate(sorted_core_pos):
        if i == 0:
            cores[pos].append((pos-2)-(previous_pos))
        else:
            cores[pos].append((pos-2)-(previous_pos)-1)
        previous_pos = pos+5
    return cores


# Function to loop through the identifed cores and find the closes match to a binding site in the OLS library using the hamming distance
def findClosestOLSMatch(tfbs_dict, match_dict):
    for pos, bs in tfbs_dict.items():
        seq = bs[2]
        closest_match = None
        min_distance = np.inf
        for key, val in match_dict.items():
            if key == "G2F":
                dist = _hamming_distance(seq[1:], match_dict[key])
            elif key == "G2R":
                dist = _hamming_distance(seq[:-1], match_dict[key])
            else:
                dist = _hamming_distance(seq, match_dict[key])
            if dist < min_distance:
                min_distance = dist
                closest_match = key
        tfbs_dict[pos].append(closest_match)
        tfbs_dict[pos].append(match_dict[closest_match])
        tfbs_dict[pos].append(min_distance)
    return tfbs_dict


# Collates a TFBS dictionary for a given input sequence. Assumes there are GATA and ETS cores to find
def defineTFBS(seq, findClosestOLS=True):
    tfbs = findEtsAndGataCores(seq)
    tfbs = findTFBSAffinity(seq, tfbs)
    tfbs = findSpacingBetweenTFBS(tfbs)
    if findClosestOLS:
        OLS_dict=loadSiteName2bindingSiteSequence()
        findClosestOLSMatch(tfbs, OLS_dict)
    return tfbs


def convert2pyRangesDict(names, seqs):
    """Convert a list of names and sequences to a dictionary of pyRanges objects"""
    d = {"Chromosome": [], "Start": [], "End": [], "Strand": [], "Name": []}
    for i, seq in tqdm(enumerate(seqs)):
        tfbs_def = defineTFBS(seq)
        name = names[i]
        for key in tfbs_def.keys():
            d["Chromosome"].append(name)
            d["Start"].append(int(key-2))
            d["End"].append(int(key-2) + len(tfbs_def[key][2])-1)
            d["Strand"].append("+" if tfbs_def[key][1] == "F" else "-")
            d["Name"].append(tfbs_def[key][0])
    return d


def convert2pyRanges(names, seqs):
    """Convert a list of names and sequences to a list of pyRanges objects"""
    import pyranges as pr
    d = convert2pyRangesDict(names, seqs)
    return pr.from_dict(d)


# Function to encode a sequence based on nucleotides. Makes use of defineTFBS to find TFBS. Note that the current
# implementation only keeps sequences with exactly 5 binding sites. Currently supports encoding into mixed 1.0, 2.0 and 3.0
def encode_seq(seq, encoding):
    if encoding not in ["mixed1", "mixed2", "mixed3"]:
        raise ValueError("Specified encoding not supported")
    enh_tfbs = defineTFBS(seq)
    if len(enh_tfbs) != 5:
        return -1
    enh_encoding = []
    for pos, tfbs in enh_tfbs.items():
        if tfbs[0] == "ETS":
            if encoding == "mixed1":
                enh_encoding += [tfbs[4], "E", tfbs[1], tfbs[3]]
            elif encoding == "mixed2":
                enh_encoding += [tfbs[4], tfbs[3], tfbs[1], 0, 0]
            elif encoding == "mixed3":
                enh_encoding += [tfbs[4], tfbs[3], 0, tfbs[1]]
        elif tfbs[0] == "GATA":
            if encoding == "mixed1":
                enh_encoding += [tfbs[4], "G", tfbs[1], tfbs[3]]
            elif encoding == "mixed2":
                enh_encoding += [tfbs[4], 0, 0, tfbs[3], tfbs[1]]
            elif encoding == "mixed3":
                enh_encoding += [tfbs[4], 0, tfbs[3], tfbs[1]]
    enh_encoding.append(len(seq)-(pos+5)-1)
    return enh_encoding


# Uses encode_seq to encode an entire dataset (an input dataframe with a column containing the sequence called "SEQ")
# Currently supports encoding into mixed 1.0, 2.0 and 3.0
def encode_dataset(data, encode):
    if encode not in ["mixed1", "mixed2", "mixed3"]:
        raise ValueError("Specified encoding not supported")
    mixed_encoding, valid_idx = [], []
    for i, (row_num, enh_data) in tqdm.tqdm(enumerate(data.iterrows())):
        sequence = enh_data["SEQ"].upper().strip()
        encoded_seq = encode_seq(sequence, encoding=encode)
        if encoded_seq != -1:
            mixed_encoding.append(encoded_seq)
            valid_idx.append(i)
    if encode in ["mixed1", "mixed3"]:
        X_mixed = (pd.DataFrame(mixed_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    elif encode == "mixed2":
        X_mixed = (pd.DataFrame(mixed_encoding).replace({"R": -1, "F": 1})).values
    if X_mixed.shape[1] not in [21,26]:
        print("Get in shape! Should have 21 or 26 features, but have {}".format(X_mixed.shape[1]))
        return -1
    return X_mixed, valid_idx


# Function to encode a sequence based on sitenames (i.e. S1-G1r...)
def encode_OLS_seq(OLS_seq, encoding, sitename_dict, affinity_dict):
    if encoding not in ["mixed1", "mixed2", "mixed3"]:
        raise ValueError("Specified encoding not supported")

    enh_enc = []  # Single enhancer encoding

    # Loop through each position
    for col_num in range(len(OLS_seq)):
        # If we have a spacer in the current position we need to check for surrounding GATA-2 sites
        if "S" in OLS_seq[col_num]:

            # If the spacer is the empty spacer, just add a 0 and go to the next
            if OLS_seq[col_num] == "S5":
                enh_enc.append(len(sitename_dict[OLS_seq[col_num]]))
                continue

            # If the spacer is downstream of a GATA-2 reverse, we need to add a nucleotide to the GATA-2 (subtract one from spacer)
            if col_num > 0:
                if OLS_seq[col_num - 1] == "G2R":
                    enh_enc.append(len(sitename_dict[OLS_seq[col_num]]) - 1)
                    continue

            # If the spacer is upstream of a GATA-2 forward, we need to add a nucleotide to the GATA-2 (subtract one from spacer)
            if col_num < len(OLS_seq) - 1:
                if OLS_seq[col_num + 1] == "G2F":
                    enh_enc.append(len(sitename_dict[OLS_seq[col_num]]) - 1)
                    continue

            # Finally if no G2F or S5 is involved, just add the normal len of the spacer
            enh_enc.append(len(sitename_dict[OLS_seq[col_num]]))
            continue

        # If we are at a TFBS, add the TFBS type, orientation and affinity
        else:
            tfbs = OLS_seq[col_num]
            tf = tfbs[0]
            aff = affinity_dict[tfbs[:2]]
            orient = tfbs[2]
            if encoding == "mixed1":
                enh_enc += [tf, orient, aff]
            elif encoding == "mixed2":
                if tf == "E":
                    enh_enc += [aff, orient, 0, 0]
                elif tf == "G":
                    enh_enc += [0, 0, aff, orient]
            elif encoding == "mixed3":
                if tf == "E":
                    enh_enc += [aff, 0, orient]
                elif tf == "G":
                    enh_enc += [0, aff, orient]

    return enh_enc


# Function to encode a dataset of OLS sequences by looping through dataframe
# and repeatedly running encode_OLS_seq. Supports mixed 1.0, 2.0 and 3.0
def encode_OLS_dataset(OLS_dataset, encode):
    site_dict = loadSiteName2bindingSiteSequence()  # Sitenames to sequence
    aff_dict = loadBindingSiteName2affinities()  # Sitenames to affinities
    if encode not in ["mixed1", "mixed2", "mixed3"]:
        raise ValueError("Specified encoding not supported")
    mixed_encoding = []
    for i, (row_num, enh_data) in enumerate(tqdm.tqdm(OLS_dataset.iterrows())):
        sequence = enh_data.values
        encoded_seq = encode_OLS_seq(sequence, encode, site_dict, aff_dict)
        mixed_encoding.append(encoded_seq)
    if encode in ["mixed1", "mixed3"]:
        X_mixed = (pd.DataFrame(mixed_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    elif encode == "mixed2":
        X_mixed = (pd.DataFrame(mixed_encoding).replace({"R": -1, "F": 1})).values
    if X_mixed.shape[1] not in [21,26]:
        print("Get in shape! Should have 21 or 26 features, but have {}".format(X_mixed.shape[1]))
        return -1
    return X_mixed


# Function to plot genome tracks for otxa
def otxGenomeTracks(seq, importance_scores=None, model_pred=None, seq_name=None, threshold=0.5, highlight=[], cmap=None, norm=None):
    # Get the annotations for the seq
    tfbs_annot = defineTFBS(seq)

    # Define subplots
    fig, ax = plt.subplots(2, 1, figsize=(12,4), sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)

    # Build the annotations in the first subplot
    h = 0.1  # height of TFBS rectangles
    ax[0].set_ylim(0, 1)  # lims of axis
    ax[0].spines['bottom'].set_visible(False)  #remove axis surrounding, makes it cleaner
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].tick_params(left = False)  #remove tick marks on y-axis
    ax[0].set_yticks([0.25, 0.525, 0.75])  # Add ticklabel positions
    ax[0].set_yticklabels(["TFBS", "Affinity", "Closest OLS Hamming Distance"], weight="bold")  # Add ticklabels
    ax[0].hlines(0.2, 1, len(seq), color="black")  #  Backbone to plot boxes on top of

    # Build rectangles for each TFBS into a dictionary
    tfbs_blocks = {}
    for pos in tfbs_annot.keys():
        if tfbs_annot[pos][0] == "GATA":
            tfbs_blocks[pos] = mpl.patches.Rectangle((pos-2, 0.2-(h/2)), width=8, height=h, facecolor="orange", edgecolor="black")
        elif tfbs_annot[pos][0] == "ETS":
            tfbs_blocks[pos] = mpl.patches.Rectangle((pos-2, 0.2-(h/2)), width=8, height=h, facecolor="blue", edgecolor="black")

    # Plot the TFBS with annotations, should be input into function
    for pos, r in tfbs_blocks.items():
        ax[0].add_artist(r)
        rx, ry = r.get_xy()
        ytop = ry + r.get_height()
        cx = rx + r.get_width()/2.0
        tfbs_site = tfbs_annot[pos][0] + tfbs_annot[pos][1]
        tfbs_aff = round(tfbs_annot[pos][3], 2)
        closest_match = tfbs_annot[pos][5] + ": " + str(tfbs_annot[pos][7])
        spacing = tfbs_annot[pos][4]
        ax[0].annotate(tfbs_site, (cx, ytop), color='black', weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(tfbs_aff, (cx, 0.45), color=r.get_facecolor(), weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(closest_match, (cx, 0.65), color="black", weight='bold',
                    fontsize=12, ha='center', va='bottom')
        ax[0].annotate(str(spacing), (((rx-spacing) + rx)/2, 0.25), weight='bold', color="black",
                fontsize=12, ha='center', va='bottom')

    if importance_scores is None:
        print("No importance scores given, outputting just sequence")
        ylab = "Sequence"
        ax[1].spines['left'].set_visible(False)
        ax[1].set_yticklabels([])
        ax[1].set_yticks([])
        importance_scores = ohe_seq(seq)
    else:
        ylab = "Importance Score"

    title = ""
    if seq_name is not None:
        title += seq_name
    if model_pred is not None:
        color = cmap(norm(model_pred))
        title += ": {}".format(str(round(model_pred, 3)))
    else:
        color = "black"

    # Plot the featue importance scores
    if len(highlight) > 0:
        to_highlight = {"red": _collapse_pos(highlight)}
        print(to_highlight)
        viz_sequence.plot_weights_given_ax(ax[1], importance_scores, subticks_frequency=10, highlight=to_highlight, height_padding_factor=1)
    else:
        viz_sequence.plot_weights_given_ax(ax[1], importance_scores, subticks_frequency=10, height_padding_factor=1)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_xlabel("Sequence Position")
    ax[1].set_ylabel(ylab)
    #ax[1].hlines(1, len(seq), threshold/10, color="red")
    plt.suptitle(title, fontsize=24, weight="bold", color=color)


def tile_plot(data, tile_col="TILE", score_col="SCORES", name_col="NAME", label_col=None):
    rc = {"font.size": 14}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(16,8))
        cmap = mpl.cm.RdYlGn
        ax.scatter(x=data[tile_col], y=data[name_col], c=data[score_col], cmap=cmap)
        cax = fig.add_axes([0.15, 0.0, 0.75, 0.02])
        norm = mpl.colors.Normalize(vmin=data[score_col].min(), vmax=data[score_col].max())
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label("Scores")
        ax.set_ylabel("Sequence")
        ax.set_xlabel("Start Position")
        start, end = data[tile_col].astype(int).min(), data[tile_col].astype(int).max()
        ax.xaxis.set_ticks(np.arange(start, end, 10))
        ax.set_xticklabels(np.arange(start, end, 10))
        if label_col != None:
            red_patch = mpatches.Patch(color='lightgreen', label='Active')
            green_patch = mpatches.Patch(color='lightcoral', label='Inactive')
            legend = plt.legend(title='Validated Status', handles=[green_patch, red_patch], bbox_to_anchor=(-0.25,0))
            plt.gca().add_artist(legend)
            for label in ax.get_yticklabels():
                print(label)
                #if data.set_index(name_col).loc[label.get_text()][label_col] == 1:
                #    label.set_color("lightgreen")
                #elif data.set_index(name_col).loc[label.get_text()][label_col] == 0:
                #    label.set_color("lightcoral")


# Wrapper function to generate mixed 1.0-3.0 encodings
# Currently supports encoding into mixed 1.0, 2.0 and 3.0
def mixed_encode(data):
    mixed1_encoding, mixed2_encoding, mixed3_encoding, valid_idx = [], [], [], []
    for i, (row_num, enh_data) in tqdm.tqdm(enumerate(data.iterrows())):
        sequence = enh_data["SEQ"].upper().strip()
        encoded_seq1 = encode_seq(sequence, encoding="mixed1")
        encoded_seq2 = encode_seq(sequence, encoding="mixed2")
        encoded_seq3 = encode_seq(sequence, encoding="mixed3")
        if encoded_seq1 != -1:
            mixed1_encoding.append(encoded_seq1)
            mixed2_encoding.append(encoded_seq2)
            mixed3_encoding.append(encoded_seq3)
            valid_idx.append(i)
    X_mixed1 = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2 = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3 = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1, X_mixed2, X_mixed3, valid_idx


# Wrapper function to encode all three mixed encodings for the OLS library. \
# Currrently supports mixed 1.0, 2.0 and 3.0
def mixed_OLS_encode(OLS_dataset):
    site_dict = loadSiteName2bindingSiteSequence()  # Sitenames to sequence
    aff_dict = loadBindingSiteName2affinities()  # Sitenames to affinities
    mixed1_encoding, mixed2_encoding, mixed3_encoding = [], [], []
    for i, (row_num, enh_data) in enumerate(tqdm.tqdm(OLS_dataset.iterrows())):
        sequence = enh_data.values
        encoded_seq1 = encode_OLS_seq(sequence, encoding="mixed1", sitename_dict=site_dict, affinity_dict=aff_dict)
        encoded_seq2 = encode_OLS_seq(sequence, encoding="mixed2", sitename_dict=site_dict, affinity_dict=aff_dict)
        encoded_seq3 = encode_OLS_seq(sequence, encoding="mixed3", sitename_dict=site_dict, affinity_dict=aff_dict)
        mixed1_encoding.append(encoded_seq1)
        mixed2_encoding.append(encoded_seq2)
        mixed3_encoding.append(encoded_seq3)
    X_mixed1s = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2s = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3s = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1s, X_mixed2s, X_mixed3s


# Wrapper function to encode all three mixed encodings for the OLS library. \
# Currrently supports mixed 1.0, 2.0 and 3.0
def otx_encode(seqs):
    mixed1_encoding, mixed2_encoding, mixed3_encoding, valid_idx = [], [], [], []
    for i, sequence in tqdm.tqdm(enumerate(seqs)):
        encoded_seq1 = encode_seq(sequence, encoding="mixed1")
        encoded_seq2 = encode_seq(sequence, encoding="mixed2")
        encoded_seq3 = encode_seq(sequence, encoding="mixed3")
        if encoded_seq1 != -1:
            mixed1_encoding.append(encoded_seq1)
            mixed2_encoding.append(encoded_seq2)
            mixed3_encoding.append(encoded_seq3)
            valid_idx.append(i)
    X_mixed1 = (pd.DataFrame(mixed1_encoding).replace({"G": 0, "E": 1, "R": 0, "F": 1})).values
    X_mixed2 = (pd.DataFrame(mixed2_encoding).replace({"R": -1, "F": 1})).values
    X_mixed3 = (pd.DataFrame(mixed3_encoding).replace({"R": 0, "F": 1})).values
    return X_mixed1, X_mixed2, X_mixed3, valid_idx
