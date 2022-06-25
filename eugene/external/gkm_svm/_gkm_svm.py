import numpy as np

# Function to grab scores from output of gkmtest
# name => filepath to read from
def get_scores(fname):
    f = open(fname)
    d = np.array([float(x.strip().split('\t')[1]) for x in f])
    f.close()
    return d
