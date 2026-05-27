import os
import kagglehub
import wfdb

def get_data_path(local_folder="Training_WFDB"):
    
    if os.path.exists(local_folder) and len(os.listdir(local_folder)) > 0:
        return local_folder
    
    cache_path = kagglehub.dataset_download("physionet/china-physiological-signal-challenge-in-2018")        
    return cache_path

def load_all_metadata(data_dir):
    patients = []
    
    all_files = os.listdir(data_dir)
    
    for filename in all_files:
        if filename.endswith('.hea'):
            record_name = filename[:-4]
            header = wfdb.rdheader(os.path.join(data_dir, record_name))
            
            patient = {'id': record_name}
            
            for comment in header.comments:
                key, value = comment.split(':')
                patient[key.strip()] = value.strip()
            
            patients.append(patient)
    
    return patients