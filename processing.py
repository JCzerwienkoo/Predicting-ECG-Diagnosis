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

RARE_LABELS = [
    "164931005",
    "284470004",
    "426783006",
    "429622005"   
]

def contains_rare_labels(labels: list):
    for l in RARE_LABELS:
        if labels.count(l) > 0:
            return True    
    return False

def generate_partitions(length, fs, length_cap):
    max_seg = length / length_cap
    end = math.floor(max_seg) * length_cap * fs
    r = range(0, end + 1, length_cap * fs)
    
    partitions = []

    start = 0    
    for i in r[1:]:
        partitions.append((start, i))
        start = i
        
    return partitions
    

def process_entry(data_dir, record_name, split_extra_samples, length_cap):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    fs = record.fs
    
    len_s = record.sig_len / fs
    
    if len_s < length_cap:
        print(f"Skipping record {record_name} due to short length ({len_s})")
        return [] 
    
    partitions = generate_partitions(len_s, fs, length_cap)
    
    labels = comment_to_labels(record.comments[2])
    has_rare_labels = contains_rare_labels(labels)
    
    if not has_rare_labels or not split_extra_samples:
        partitions = [partitions[0]]
    
    new_entries = []
    
    for start, end in partitions:
        data = []
        for lead in range(len(record.sig_name)):
            raw_signal = record.p_signal[start:end, lead]
            b, a = signal.butter(3, [0.5, 50], btype='bandpass', fs=fs)
            filtered_signal = signal.filtfilt(b, a, raw_signal)

            _a, _b, spect = signal.spectrogram(filtered_signal, fs=fs, nperseg=32)
            
            spect_db = 10 * np.log10(spect + 1e-10)
            data.append(spect_db)
            
        
        new_entries.append({
            "record": record,
            "data": np.array(data),
            "label": labels 
        })
        
    return new_entries
    
def precompute_data(data_dir, patients, split_extra_samples, length_cap):
    entries = []
    i = 1
    
    for patient in patients:
        new_entries = process_entry(data_dir, patient['id'], split_extra_samples, length_cap)
        for entry in new_entries:
            entries.append(entry)
            
        print(f"Precomputing patient no. {i} / {len(patients)}, {len(new_entries)} entries added")
        i += 1
        
    return entries

def precompute_and_save(input_folder, output, split_extra_samples, length_cap):
    patients = load_all_metadata(input_folder)

    print("Computing data")
    precomp = precompute_data(input_folder, patients, split_extra_samples, length_cap)
    
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
    
