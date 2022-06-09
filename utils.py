from functools import reduce
import numpy as np

def extract_vector(m:np.array, index:int):
    return m[:,index]

def one_hot_encoder(size:int):
    m = np.identity(size, dtype=np.int32)
    d = dict()
    for i in range(size):
        d[i] = extract_vector(m, i)
    return d

def divide_matrix(ary:np.array, num_columns:int, num_rows:int):    
    
    if (num_columns > ary.shape[1] or num_rows > ary.shape[0]):
        return 'Error, the requested number of columns or rows exceed dimensions!'
    
    first = np.array_split(ary, num_rows, axis=0) # num rows
    second = []
    for array in first:
        second.append(np.array_split(array, num_columns, axis=1)) # num columns
    flattened_list = flatten(second)
    return flattened_list

def flatten(l:list):
    flattened_list = []
    for item in l:
        for i in item:
            flattened_list.append(i)
    return flattened_list
    
def compute_feature(list_of_arrays:list, prefix=None) -> dict:
    
    d = dict()
    
    if prefix == None:
      for id in range(len(list_of_arrays)):
          d[f'{id}_mean_feature'] = np.mean(list_of_arrays[id])
          d[f'{id}_std_feature'] = np.std(list_of_arrays[id])
    else:
      for id in range(len(list_of_arrays)):
          d[f'{id}_{prefix}_mean_feature'] = np.mean(list_of_arrays[id])
          d[f'{id}_{prefix}_std_feature'] = np.std(list_of_arrays[id])

    return d


import noisereduce as nr
import librosa
from scipy.io import wavfile

def cleaned_file(filename:str, sample_rate:int, use_librosa:bool=True):
    """return a signal cleaned with reduced noise

    Args:
        filename (str): name of the wav file to modify
        sample_rate (int): sample rate used to sample the wav file
        use_librosa (bool, optional): If True use librosa, if False use scipy.wav. Defaults to True.

    Returns:
        np.array: new signal
    """
    
    if use_librosa:
        signal, _ = librosa.load(filename, sr=sample_rate, res_type='kaiser_fast') # signal already normalized
        norm_reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate)
        return norm_reduced_noise
    else:
        _, signal = wavfile.read(filename)
        reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate)
        return reduced_noise