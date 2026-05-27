import os
import matplotlib.pyplot as plt
import wfdb
from data_loader import get_data_path
from collections import Counter

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

def plot_diagnosis_distribution(patients):
    all_dx = []
    
    for patient in patients:
        dx_codes = patient['Dx'].split(',')  
        for code in dx_codes:
            all_dx.append(code.strip())
    
    counts = Counter(all_dx)
    
    print(f"Unique diagnoses: {len(counts)}")

    for code, count in counts.most_common():
        print(f"{code} -> {count} patients")
    
    plt.figure(figsize=(14, 5))
    plt.bar(counts.keys(), counts.values(), color='steelblue')
    plt.title('Diagnosis distribution')
    plt.xlabel('Diagnosis code')
    plt.ylabel('Number of patients')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/diagnosis_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    os.makedirs('plots', exist_ok=True)
    data_dir = get_data_path()

    patients = load_all_metadata(data_dir)
    print(f"Total patients: {len(patients)}")
    print(f"First patient: {patients[0]}")

    plot_diagnosis_distribution(patients)


if __name__ == "__main__":
    main()