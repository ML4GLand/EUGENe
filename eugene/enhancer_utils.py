import re
import numpy as np
import logging
np.random.seed(42)

enhancer_binding_sites = {"Core-otx-a": ["..GGAA..", "..GGAT..", "..TTCC..", "..ATCC..", "..GATA..", "..TATC.."],
                          "WT-otx-a": ["GTTATCTC", "ACGGAAGT", "AAGGAAAT", "AATATCT", "AAGATAGG", "GAGATAAC", "ACTTCCGT", "ATTTCCTT", "AGATATT", "CCTATCTT"]}
alphabet = np.array(["A", "G", "C", "T"])


def is_overlapping(a, b):
    if b[0] >= a[0] and b[0] <= a[1]:
        return True
    else:
        return False

def merge_intervals(intervals):
    merged_list= []
    merged_list.append(intervals[0])
    for i in range(1, len(intervals)):
        pop_element = merged_list.pop()
        if is_overlapping(pop_element, intervals[i]):
            new_element = pop_element[0], max(pop_element[1], intervals[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(intervals[i])
    return merged_list

def randomizeLinkers(seq, features=None, enhancer=None):   
    if features == None:
        assert enhancer != None
        features = enhancer_binding_sites[enhancer]
        
    transformed_seq = []
    feature_spans = merge_intervals([x.span() for x in re.finditer(r"("+'|'.join(features)+r")", seq)])
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


