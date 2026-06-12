import os
import matplotlib.pyplot as plt
import wfdb
from data_loader import get_data_path
from collections import Counter
from scipy import signal
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score

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

def plot_ecg(data_dir, record_name, save=False):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    header = wfdb.rdheader(os.path.join(data_dir, record_name))

    info = {}
    for comment in header.comments:
        key, value = comment.split(':')
        info[key.strip()] = value.strip()

    for fig_num in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i in range(4):
            row = i // 2
            col = i % 2
            lead_index = fig_num * 4 + i
            signal = record.p_signal[:, lead_index]
            lead_name = record.sig_name[lead_index]

            axes[row][col].plot(signal, linewidth=0.8, color='steelblue')
            axes[row][col].set_title(lead_name)
            axes[row][col].grid(True, alpha=0.3)

        plt.suptitle(
            f'ECG: {record_name} | Age: {info.get("Age","?")} | '
            f'Sex: {info.get("Sex","?")} | Dx: {info.get("Dx","?")} | '
            f'Part {fig_num + 1}/3'
        )
        plt.tight_layout()

        if save:
            plt.savefig(f'plots/ecg_{record_name}_part{fig_num+1}.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_ecg_spectrogram(data_dir, record_name, save=False):
    """
    Converts 1D ECG time-series data from Lead II into a 2D Spectrogram image.
    """
    
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    fs = record.fs  
    
    lead_index = 1
    raw_signal = record.p_signal[:, lead_index]
    lead_name = record.sig_name[lead_index]

    b, a = signal.butter(3, [0.5, 45], btype='bandpass', fs=fs)
    filtered_signal = signal.filtfilt(b, a, raw_signal)


    frequencies, times, spect = signal.spectrogram(filtered_signal, fs=fs, nperseg=32)
    
    spect_db = 10 * np.log10(spect + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, spect_db, shading='gouraud', cmap='viridis')
    
    plt.title(f'2D Spectrogram (Time-Frequency) - Lead: {lead_name} | Patient: {record_name}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seconds]')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()

    if save:
        plt.savefig(f'plots/spectrogram_{record_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_multilabel_evaluation(y_true, y_pred, label_names, save=False, output_prefix='plots/multilabel_eval'):
    """Plot multilabel evaluation charts that are readable for many label combinations."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError('y_true and y_pred must be 2D arrays of shape (n_samples, n_labels)')

    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')

    n_labels = y_true.shape[1]
    if len(label_names) != n_labels:
        raise ValueError('label_names length must match number of labels')

    if save:
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    matrices = multilabel_confusion_matrix(y_true, y_pred)

    cols = min(3, n_labels)
    rows = int(np.ceil(n_labels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if idx >= n_labels:
            ax.axis('off')
            continue

        cm = matrices[idx]
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1], ['Predict False', 'Predict True'])
        ax.set_yticks([0, 1], ['Actual False', 'Actual True'])
        ax.set_title(str(label_names[idx]))

        max_val = np.max(cm)
        threshold = max_val / 2.0 if max_val > 0 else 0.0
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > threshold else 'black'
                ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color=color)

    plt.suptitle('Per-label confusion matrices', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(f'{output_prefix}_confusion_grid.png', dpi=180, bbox_inches='tight')
    plt.show()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    prevalence = y_true.mean(axis=0)

    metric_matrix = np.vstack([precision, recall, f1, prevalence])
    metric_names = ['Precision', 'Recall', 'F1', 'Prevalence']

    plt.figure(figsize=(max(8, n_labels * 1.2), 4.5))
    plt.imshow(metric_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    plt.xticks(range(n_labels), label_names, rotation=45, ha='right')
    plt.yticks(range(len(metric_names)), metric_names)
    plt.colorbar(label='Score')
    plt.title('Per-label metric heatmap')

    for i in range(metric_matrix.shape[0]):
        for j in range(metric_matrix.shape[1]):
            val = metric_matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(f'{output_prefix}_metric_heatmap.png', dpi=180, bbox_inches='tight')
    plt.show()

    global_metrics = {
        'Subset Acc': accuracy_score(y_true, y_pred),
        'Micro F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Samples F1': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'Jaccard': jaccard_score(y_true, y_pred, average='samples', zero_division=0),
        'Hamming': 1.0 - hamming_loss(y_true, y_pred),
    }

    labels = list(global_metrics.keys())
    values = list(global_metrics.values())

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, values, color=['#386cb0', '#7fc97f', '#fdc086', '#beaed4', '#f0027f', '#bf5b17'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Global multilabel metrics')
    plt.xticks(rotation=25, ha='right')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(f'{output_prefix}_global_metrics.png', dpi=180, bbox_inches='tight')
    plt.show()