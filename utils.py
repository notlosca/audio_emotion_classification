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


##----Not used since didn't improve results----##
#
import noisereduce as nr
import librosa
from scipy.io import wavfile
#
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
        return deal_nan_val(norm_reduced_noise)
    else:
        _, signal = wavfile.read(filename)
        reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate)
        return deal_nan_val(reduced_noise)
#    
def deal_nan_val(ary:np.array) -> np.array:
    """ If one entry of the cleaned signal is nan, it is replaced with 0"""
    if np.sum(np.isnan(ary)) != 0:
        idxs = np.argwhere(np.isnan(ary)==True).reshape(-1,)
        for id in idxs:
            ary[id] = 0.
    return ary
#
##----Not used since didn't improve results----##


## PCA selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt




def run_PCA_best_n(list_of_n:list, X_res, y_res):
    scores_rf = []
    scores_svm = [] 
    scores_knn = []
    for n in tqdm(list_of_n):
        print(n)
        minmax = MinMaxScaler()
        norm_X_res = minmax.fit_transform(X_res)
        
        pca = PCA(n_components=n, random_state=42)
        pca_X_res = pca.fit_transform(norm_X_res)



        X_train, X_test, y_train, y_test = train_test_split(pca_X_res, y_res, test_size=.2, random_state=42)
        
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        print('Fitting RF...')
        rf = RandomForestClassifier().fit(X_train, y_train) # plain rf
        print('Predicting using RF...')
        preds = rf.predict(X_test)
        score = f1_score(y_test, preds, average='macro')
        scores_rf.append(score)
        print(rf, score)

        print('Fitting KNN...')
        knn = KNeighborsClassifier().fit(X_train, y_train) # plain knn
        print('Predicting using KNN...')
        preds = knn.predict(X_test)
        score = f1_score(y_test, preds, average='macro')
        scores_knn.append(score)
        print(knn, score)

        print('Fitting SVC...')
        svm = SVC().fit(X_train, y_train)
        print('Predicting using SVC...')
        svm_preds = svm.predict(X_test)
        svm_score = f1_score(y_test, svm_preds, average='macro')
        scores_svm.append(svm_score)
        print(svm, svm_score)
        
    return scores_rf, scores_svm, scores_knn

def plot_pca_selection(list_of_n, scores_rf, scores_svm, scores_knn):
    plt.figure(figsize=(12,10))
    plt.plot(list_of_n, scores_rf, label='plain RF')
    plt.plot(list_of_n, scores_svm, label='plain SVM')
    plt.plot(list_of_n, scores_knn, label='plain KNN')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('F1 macro')
    plt.title('F1 macro score as number of PCA components varies')
    

from sklearn.model_selection import ParameterGrid

def run_svm_grid_search(param_grid:dict, svm_X_train, svm_X_test, y_train, y_test):
    clfs = []
    for configuration in tqdm(ParameterGrid(param_grid)):
        print('Fitting SVC...')
        svm = SVC(random_state=42, **configuration).fit(svm_X_train, y_train)
        print('Predicting using SVC...')
        svm_preds = svm.predict(svm_X_test)
        svm_score = f1_score(y_test, svm_preds, average='macro')
        print(svm, svm_score)
        clfs.append((svm, svm_score))
    return clfs



########################################################

# functions to process and visualize data

import noisereduce as nr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_file(audio_file, normalized=True) ->  tuple:
    if normalized:
        signal, sr = librosa.load(audio_file, sr=None, res_type='kaiser_fast', mono=True)
    else:
        sr, signal = wavfile.read(audio_file)
    return signal, sr

def is_normalized(signal):
    if max(np.abs(signal)) <= 1:
        return True
    else:
        return False
        
def plot_waveform(signal, sr, title, figsize=(15,10), margins=(0.01,0.01), **kwargs):
    plt.figure(figsize=figsize)
    plt.title(title)
    time_points = np.linspace(0, len(signal)/sr, len(signal))
    plt.plot(time_points, signal)
    plt.xlabel('Time (s)')
    # check wether the signal is normalized or not
    if is_normalized(signal):
        plt.ylabel('Normalized amplitude')
    else: 
        plt.ylabel('Amplitude')
    
    if margins != None:
        x = margins[0]
        y = margins[1]
        plt.margins(x=x, y=y) 

def plot_linear_magnitude_spectrum(signal, sr, filename, margins=(0.01,0.01),  **kwargs):
    plt.title(f"Linear scale of {filename}")
    plt.magnitude_spectrum(signal, sr, scale='linear', **kwargs)
    plt.xlabel('Frequency (Hz)')
    if margins != None:
        x = margins[0]
        y = margins[1]
        plt.margins(x=x, y=y) 

def plot_db_magnitude_spectrum(signal, sr, filename, margins=(0.01,0.01), **kwargs):
    plt.title(f"dB scale of {filename}")
    plt.magnitude_spectrum(signal, sr, scale='dB', **kwargs)
    plt.xlabel('Frequency (Hz)')
    if margins != None:
        x = margins[0]
        y = margins[1]
        plt.margins(x=x, y=y) 

def plot_magnitude_spectrums(signal, sr, filename, margins=(0.03,0.03), figsize=(15,10)):
    # set figure dimension
    plt.figure(figsize=figsize)
    
    # plot linear energy over frequency 
    plt.subplot(2,1,1)
    plot_linear_magnitude_spectrum(signal, sr, filename, margins)
    
    # plot log energy over frequency (as human perceive)
    plt.subplot(2,1,2)
    plot_db_magnitude_spectrum(signal, sr, filename, margins)

'''def plot_comparison(signal1, signal2, sr, dictionary, figsize=(15,10), kind='waveform'):
    """plot the comparison between the same audio, original and processed.
    Hence it will have the same sample rate (sr).

    Args:
        signal1 (_type_): _description_
        signal2 (_type_): _description_
        sr (_type_): _description_
        dictionary (_type_): dictionary containing **kwargs of each signal plot
        kind (str, optional): _description_. Defaults to 'waveform'.
    """
    
    plt.figure(figsize=figsize)
    
    d1_kwargs = dictionary['signal1']
    d2_kwargs = dictionary['signal2']
    
    if kind == 'frequency':
        plot_magnitude_spectrums(signal1, sr, d1_kwargs)
        plot_magnitude_spectrums(signal2, sr, d2_kwargs)
    else:
        plot_waveform(signal1, sr, d1_kwargs)
        plot_waveform(signal2, sr, d2_kwargs)'''
        

def denoise(signal, sr):
    return nr.reduce_noise(signal, sr)

def trim_silence(signal, top_db=30):
    trimmed_signal, index = librosa.effects.trim(signal, top_db=top_db)
    return trimmed_signal

def denoise_and_trim(signal, sr, top_db=30):
    denoised_signal = denoise(signal, sr)
    return trim_silence(denoised_signal, top_db)