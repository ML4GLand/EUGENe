import numpy as np


#https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-array
def k_largest_index_argsort(
    a: np.ndarray, 
    k: int = 1
) -> np.ndarray:
    """
    Returns the indeces of the k largest values of a numpy array. 
    
    If a is multi-dimensional, the indeces are returned as an array k x d array where d is 
    the dimension of a. The kth row represents the kth largest value of the overall array.
    The dth column returned repesents the index of the dth dimension of the kth largest value.
    So entry [i, j] in the return array represents the index of the jth dimension of the ith
    largets value in the overall array.

    a = array([[38, 14, 81, 50],
               [17, 65, 60, 24],
               [64, 73, 25, 95]])

    k_largest_index_argsort(a, k=2)
    array([[2, 3],  # first largest value is at [2,3] of the array (95)
           [0, 2]])  # second largest value is at [0,2] of the array (81)


    Parameters
    ----------
    a : numpy array
        The array to get the k largest values from.
    k : int
        The number of largest values to get.
    
    Returns
    -------
    numpy array
        The indexes of the k largest values of a.
    """
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


