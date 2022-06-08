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

