import os
import keras
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
from plot import plot_multilabel_evaluation, plot_training_history
from processing import flatten_data_linear, train_test_split, load_precomputed, precompute_and_save, process_entry, process_labels, convert_data_to_vectors, transpose_entries, trim_data_to_shortest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold
import pickle as pkl
import tensorflow as tf


SAMPLE_SVM = lambda *kwargs: SVMModel("default")
SAMPLE_CNN = lambda input_shape, num_labels: CNNModel("default", input_shape, num_labels, {
    "layers": [
        keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal()),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.15),

        keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal()),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.35),

        keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal()),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.35),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.40),
    ]
})

SAMPLE_CNN_V2 = lambda input_shape, num_labels: CNNModel("v2", input_shape, num_labels, {
    "layers": [
        keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer="L1L2"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer="L1L2"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal(), kernel_regularizer="L1L2"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.35),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu", kernel_regularizer="L1L2"),
        keras.layers.Dropout(0.40),
    ],
    "epoch_count": 20
})

SAMPLE_CNN_LTSM = lambda input_shape, num_labels: CNNModel("ltsm", input_shape, num_labels, {
    "layers": [
        keras.layers.Permute((3, 1, 2)),
        keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
        
        keras.layers.TimeDistributed(keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2))),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout2D(0.25)),

        keras.layers.TimeDistributed(keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2))),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout2D(0.25)),

        keras.layers.TimeDistributed(keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2))),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout2D(0.35)),

        keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D()),
        keras.layers.TimeDistributed(keras.layers.Flatten()),

        keras.layers.LSTM(128),
        keras.layers.Dense(128, activation="relu"),
    ],
    "epoch_count": 18
})


SAMPLE_CNN_LTSM_TIME_DEPENDENT = lambda input_shape, num_labels: CNNModel("ltsm-time", input_shape, num_labels, {
    "layers": [
        keras.layers.Permute((2, 1, 3)),
        keras.layers.TimeDistributed(keras.layers.Conv1D(64, kernel_size=(3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout1D(0.25)),

        keras.layers.TimeDistributed(keras.layers.Conv1D(128, kernel_size=(3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout1D(0.25)),

        keras.layers.TimeDistributed(keras.layers.Conv1D(256, kernel_size=(3), padding="same", activation="relu", kernel_initializer=keras.initializers.he_normal())),
        keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.SpatialDropout1D(0.35)),

        keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling1D()),
        keras.layers.TimeDistributed(keras.layers.Flatten()),

        keras.layers.LSTM(128),
        keras.layers.Dense(128, activation="relu"),
    ],
    "epoch_count": 18
})
SAMPLE_RESNET = lambda input_shape, num_labels: ResNet("default", input_shape, num_labels)

MODELS = [
    SAMPLE_CNN_LTSM_TIME_DEPENDENT
]

def precompute( split_extra_samples=None, length_cap=6):
    data_dir = get_data_path()
    name = ["precomputed"]

    if split_extra_samples:
        name.append("-split-gen")
        
    if length_cap != None:
        name.append(f"-length-{length_cap}")

    precompute_and_save(data_dir, f"./spectral_data/{''.join(name)}.pkl", split_extra_samples, length_cap)

def test_spectogram():
    data_dir = get_data_path()
    patients = load_all_metadata(data_dir)
    plot_ecg_spectrogram(data_dir, patients[1]['id'], save=True)
    

def main(dataset_name: str):
    
    data = load_precomputed(f"./spectral_data/{dataset_name}.pkl")
  
    X, Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y)
    X, Y, X_test, Y_test = train_test_split(X, Y)
    
    X = trim_data_to_shortest(X)
    
    k_fold = MultilabelStratifiedKFold(n_splits=3, shuffle=True)
    
    pairs = []
    
    for model in MODELS:
        for train_index, test_index in k_fold.split(X, Y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            
            pairs.append((X_train, y_train, X_test, y_test))
        
        best_model, best_loss = train_and_pick_best(model, pairs)
        
        best_loss
        best_model.save()
    

def test(model_name: str, postfix: str, dataset_name: str):
    data = load_precomputed(f"./spectral_data/{dataset_name}.pkl")
  
    X,Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y)
    X = trim_data_to_shortest(X)
    X, Y, X_test, Y_test = train_test_split(X, Y)
    
    model: SVMModel = None
    with open(f"./models/{model_name}.pkl", "rb") as f:
        model = pkl.load(f)

    if model.training_history is not None:
        plot_training_history(
            model.training_history,
            model_name=f'{model.model_name}-{model.variant_name}',
            save=True,
            output_prefix=f'plots/training{postfix}'
        )

    Y_pred = model.predict(X_test)

    plot_multilabel_evaluation(
        y_true=Y_test,
        y_pred=Y_pred,
        label_names=[str(c) for c in mlb.classes_],
        save=True,
        output_prefix=f'plots/test_multilabel_eval{postfix}'
    )

DATASET_NAME = "precomputedsplit-genlength-10"

if __name__ == "__main__":
    main(dataset_name = DATASET_NAME)
    # test("cnn-ltsm-time", "-ltsm-time-oversample-l10", dataset_name = DATASET_NAME)
