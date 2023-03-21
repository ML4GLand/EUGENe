import subprocess
import numpy as np

def check_memesuite_installation(path='meme'):
    """ check if MEME suite is installed """
    try:
        subprocess.check_output(['meme', '-version'])
    except:
        raise Exception('MEME suite not installed')

def meme(
    seqs, 
    output_path, 
    motif_length=8, 
    motif_number=5, 
    meme_path='meme'
):
  """ perform meme analysis """
  cmd = [meme_path, seq_path, '-dna', '-mod', 'zoops', '-nmotifs', str(motif_number), '-minw', str(motif_length), '-maxw', str(motif_length), '-oc', output_path]

def tomtom(
    motif_set, 
    db_path, 
    output_path, 
    evalue=False, 
    thresh=0.5, 
    dist='pearson', 
    png=None, 
    tomtom_path='tomtom'
):
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

def fimo(
    motif_set,  
    seqs,
    output_path, 
    thresh=0.5, 
    fimo_path='fimo'
):
  """ perform fimo analysis """
  cmd = [fimo_path,'-thresh', str(thresh), '-oc', output_path, motif_path, seq_path]
  process = subprocess
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

    def fimo():
    pass