# -*- coding: utf-8 -*-

"""
Python script for evaluating EUGENE project models
TODO: 
    1. Function to scan sequence for GATA, ETS cores
    2. Function to calc the affinity of each identified 8-mer binding site
    3. Function to identify the linker distance between each 8-mer binding site
    4. Function to calculate the hamming distance between each binding site and OLS TFBS
"""


import pickle
import numpy as np


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


# Load Otx-a binding site to affinity dictionary
def loadBindingSiteName2affinities(file="/cellar/users/aklie/projects/EUGENE/data/auxiliary/bindingSiteName2affinities.pkl", pickle_obj=True):
    if pickle_obj:
        with open(file, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        print("Only pickles at this time")
    
    
# Load Otx-a binding site name to sequence dictionary
def loadSiteName2bindingSiteSequence(file="/cellar/users/aklie/projects/EUGENE/data/auxiliary/siteName2bindingSiteSequence.pkl", pickle_obj=True):
    if pickle_obj:
        with open(file, 'rb') as handle:
            b = pickle.load(handle)
        return b
    else:
        print("Only pickles at this time")


# Function to return a dictionary with the position of the first nucleotide of every GATA and ETS core fond in an input sequence. Also includes orientation
def findEtsAndGataCores(seq, cores={"ETS_FORWARD": ["GGAA","GGAT"], "ETS_REVERSE": ["TTCC", "ATCC"], "GATA_FORWARD": ["GATA"], "GATA_REVERSE": ["TATC"]}):
    core_pos = {}
    for i in range(len(seq)):
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
def findTFBSAffinity(seq, cores, ets_aff_file="/cellar/users/aklie/projects/EUGENE/data/auxiliary/parsed_Ets1_8mers.txt", gata_aff_file="/cellar/users/aklie/projects/EUGENE/data/auxiliary/parsed_Gata6_3769_contig8mers.txt"):
    ets_aff = loadEtsAff(ets_aff_file)
    gata_aff = loadGata6Aff(gata_aff_file)
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


# Find hamming distance between two strings. Returns inf if they are different lengths
def hamming_distance(string1, string2): 
    distance = 0
    L = len(string1)
    if L != len(string2):
        return np.inf
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


# Function to loop through the identifed cores and find the closes match to a binding site in the OLS library using the hamming distance
def findClosestOLSMatch(tfbs_dict, match_dict):
    for pos, bs in tfbs_dict.items():
        seq = bs[2]
        closest_match = None
        min_distance = np.inf
        for key, val in match_dict.items():
            if key == "G2F":
                dist = hamming_distance(seq[1:], match_dict[key])
            elif key == "G2R":
                dist = hamming_distance(seq[:-1], match_dict[key])
            else:
                dist = hamming_distance(seq, match_dict[key])
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


