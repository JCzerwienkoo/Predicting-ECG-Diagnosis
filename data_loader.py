import os
import kagglehub

def get_data_path(local_folder="Training_WFDB"):
    
    if os.path.exists(local_folder) and len(os.listdir(local_folder)) > 0:
        return local_folder
    
    cache_path = kagglehub.dataset_download("physionet/china-physiological-signal-challenge-in-2018")        
    return cache_path