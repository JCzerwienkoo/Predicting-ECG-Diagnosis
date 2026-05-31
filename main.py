import os
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
from data_loader import get_data_path
from data_loader import load_all_metadata
from plot import plot_diagnosis_distribution
from plot import plot_demographics
from plot import plot_ecg
from plot import plot_ecg_spectrogram

def main():
    os.makedirs('plots', exist_ok=True)
    data_dir = get_data_path()

    patients = load_all_metadata(data_dir)
    print(f"Total patients: {len(patients)}")
    print(f"First patient: {patients[0]}")

    #plot_diagnosis_distribution(patients)
    #plot_demographics(patients)    

    #plot_ecg(data_dir, patients[0]['id'], save=True)

    print("Generating 2D Spectrogram for the first patient...")
    plot_ecg_spectrogram(data_dir, patients[0]['id'], save=True)


if __name__ == "__main__":
    main()