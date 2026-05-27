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

def plot_demographics(patients):
    ages = []
    sexes = []
    
    for patient in patients:
        if patient['Age'].isdigit():
            ages.append(int(patient['Age']))
        sexes.append(patient['Sex'])
    print()
    print("Age groups:")
    print(f"0-20:   {sum(1 for a in ages if a <= 20)}")
    print(f"21-40:  {sum(1 for a in ages if 21 <= a <= 40)}")
    print(f"41-60:  {sum(1 for a in ages if 41 <= a <= 60)}")
    print(f"61-80:  {sum(1 for a in ages if 61 <= a <= 80)}")
    print(f"81+:    {sum(1 for a in ages if a >= 81)}")
    print(f"Average age: {sum(ages) / len(ages):.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1.hist(ages, bins=20, color='steelblue', edgecolor='black')
    ax1.set_title('Age distribution')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Number of patients')
    ax1.grid(True, alpha=0.3)
    
    counts = Counter(sexes)
    ax2.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
    ax2.set_title('Sex distribution')
    
    plt.suptitle('Patient demographics', fontsize=13)
    plt.tight_layout()
    plt.savefig('plots/demographics.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    os.makedirs('plots', exist_ok=True)
    data_dir = get_data_path()

    patients = load_all_metadata(data_dir)
    print(f"Total patients: {len(patients)}")
    print(f"First patient: {patients[0]}")

    plot_diagnosis_distribution(patients)
    plot_demographics(patients)

    


if __name__ == "__main__":
    main()