import math
import os
import matplotlib.pyplot as plt
import wfdb
from data_loader import get_data_path, load_all_metadata
from collections import Counter
from scipy import signal
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pickle as pkl

# labels = {
#     "59118001": 1,
#     "164889003": 2,
#     "426783006": 3,
#     "429622005": 4,
#     "270492004": 5,
#     "164884008": 6,
#     "284470004": 7,
#     "164909002": 8,
#     "164931005": 9,
# }

def comment_to_labels(l: str): 
    l = l.replace("Dx: ", "")
    
    if l.count(",") == 0:
        return [l]
    
    return l.split(",")


def process_entry(data_dir, record_name):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    fs = record.fs
    data = []
    
    
    for lead in range(len(record.sig_name)):
        raw_signal = record.p_signal[:, lead]
        b, a = signal.butter(3, [0.5, 50], btype='bandpass', fs=fs)
        filtered_signal = signal.filtfilt(b, a, raw_signal)

        _a, _b, spect = signal.spectrogram(filtered_signal, fs=fs, nperseg=32)
        
        spect_db = 10 * np.log10(spect + 1e-10)
        data.append(spect_db)
        
        
    return {
        "record": record,
        "data": np.array(data),
        "label": comment_to_labels(record.comments[2])
    }
    
def precompute_data(data_dir, patients):
    entries = []
    i = 1
    
    for patient in patients:
        entries.append(process_entry(data_dir, patient['id']))
        print(f"Precomputing patient no. {i} / {len(patients)}")
        i += 1
        
    return entries

def precompute_and_save(input_folder, output):
    patients = load_all_metadata(input_folder)

    print("Computing data")
    precomp = precompute_data(input_folder, patients)
    
    print("Writing pickle")
    with open(output, "wb") as f:
        pkl.dump(precomp, f)
    print("pickling done")
    
    
def load_precomputed(input):
    data = None
    with open(input, "rb") as f:
        data = pkl.load(f)
    return data

def convert_data_to_vectors(data):
    return [r['data'] for r in data], [r['label'] for r in data]

def process_labels(Y):
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(Y), mlb

def flatten_data_linear(data):
    return np.array([np.reshape(x, (-1, 1)) for x in data])

def transpose_entries(data):
    return np.array([x.T for x in data])
    
def trim_data_to_shortest(data):
    shortest = math.inf
    for d in data:
        length = d.shape[2]
        
        if length < shortest:
            shortest = length
    
    return np.array([i[:, :, :shortest] for i in data])
    

def train_test_split(X, Y,  ratio = 0.8):
    split_point = int(len(X) * ratio)
    return X[:split_point], Y[:split_point], X[split_point:], Y[split_point:]
    
