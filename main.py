import os
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
from Model import CNNModel, ResNet, SVMModel, train_and_pick_best
from data_loader import get_data_path
from data_loader import load_all_metadata
from plot import plot_diagnosis_distribution
from plot import plot_demographics
from plot import plot_ecg
from plot import plot_ecg_spectrogram
from plot import plot_multilabel_evaluation
from processing import flatten_data_linear, train_test_split, load_precomputed, precompute_and_save, process_entry, process_labels, convert_data_to_vectors, transpose_entries, trim_data_to_shortest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold
import pickle as pkl

SAMPLE_SVM = lambda *kwargs: SVMModel("default")
SAMPLE_CNN = lambda input_shape, num_labels: CNNModel("default", input_shape, num_labels)
SAMPLE_RESNET = lambda input_shape, num_labels: ResNet("default", input_shape, num_labels)

MODELS = [
    SAMPLE_CNN
]

def precompute():
    data_dir = get_data_path()
    precompute_and_save(data_dir, "./spectral_data/precomputed.pkl")

def test_spectogram():
    data_dir = get_data_path()
    patients = load_all_metadata(data_dir)
    plot_ecg_spectrogram(data_dir, patients[1]['id'], save=True)
    

def main():
    # os.makedirs('plots', exist_ok=True)
    
    
    # data_dir = get_data_path()

    # patients = load_all_metadata(data_dir)
    # print(f"Total patients: {len(patients)}")
    # print(f"First patient: {patients[0]}")

    #plot_diagnosis_distribution(patients)
    #plot_demographics(patients)    

    # plot_ecg_spectrogram(data_dir, patients[3]['id'], save=True)

    # print("Generating 2D Spectrogram for the first patient...")
    # data = process_entry(data_dir, patients[0]['id'])
    
    # precompute_and_save(data_dir, "./spectral_data/precomputed.pkl")
    
    # return 
    data = load_precomputed("./spectral_data/precomputed.pkl")
  
    X, Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y)
    X, Y, X_test, Y_test = train_test_split(X, Y)
    
    X = trim_data_to_shortest(X)
    
    k_fold = MultilabelStratifiedKFold(n_splits=3, shuffle=True)
    
    pairs = []
    
    for train_index, test_index in k_fold.split(X, Y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        pairs.append((X_train, y_train, X_test, y_test))
    
    best_model, best_loss = train_and_pick_best(MODELS[0], pairs)
    
    best_loss
    best_model.save()
    

def test():
    data = load_precomputed("./spectral_data/precomputed.pkl")
  
    X,Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y)
    X = trim_data_to_shortest(X)
    X, Y, X_test, Y_test = train_test_split(X, Y)
    
    
    model: SVMModel = None
    with open("./models/cnn-default.pkl", "rb") as f:
        model = pkl.load(f)
        
    Y_pred = model.predict(X_test)

    plot_multilabel_evaluation(
        y_true=Y_test,
        y_pred=Y_pred,
        label_names=[str(c) for c in mlb.classes_],
        save=True,
        output_prefix='plots/test_multilabel_eval'
    )




if __name__ == "__main__":
    # precompute()
    
    main()
    test()