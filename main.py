from Model import CNNModel, ResNet, SVMModel, train_and_pick_best, train_final_model, score_multilabel_model
from data_loader import get_data_path
from data_loader import load_all_metadata
from plot import plot_ecg_spectrogram
from plot import plot_multilabel_evaluation
from processing import flatten_data_linear, train_test_split, load_precomputed, precompute_and_save, process_labels, convert_data_to_vectors, transpose_entries, resize_data_to_length
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import pickle as pkl
import math
import numpy as np

SAMPLE_SVM = lambda input_shape, num_labels: SVMModel("default")
SAMPLE_CNN = lambda input_shape, num_labels: CNNModel("default", input_shape, num_labels)
SAMPLE_RESNET = lambda input_shape, num_labels: ResNet("default", input_shape, num_labels)

MODELS = [
    {
        "name": "svm",
        "factory": SAMPLE_SVM,
        "input_type": "flat",
    },
    {
        "name": "cnn",
        "factory": SAMPLE_CNN,
        "input_type": "spectrogram",
    },
    {
        "name": "resnet",
        "factory": SAMPLE_RESNET,
        "input_type": "spectrogram",
    },
]

SAMPLE_SIZE = 1000 #None if we want to use them all 
CV_SPLITS = 2 if SAMPLE_SIZE else 3
RANDOM_STATE = 42
BEST_MODEL_PATH = "./models/best-model.pkl"

def precompute():
    data_dir = get_data_path()
    precompute_and_save(data_dir, "./spectral_data/precomputed.pkl")

def test_spectogram():
    data_dir = get_data_path()
    patients = load_all_metadata(data_dir)
    plot_ecg_spectrogram(data_dir, patients[1]['id'], save=True)
    
def get_shortest_length(X):
    return min(x.shape[2] for x in X)

def prepare_model_input(X, input_type, target_length=None):
    if target_length is None:
        target_length = get_shortest_length(X)

    X = resize_data_to_length(X, target_length)

    if input_type == "flat":
        return flatten_data_linear(X), target_length

    return transpose_entries(X), target_length

def limit_dataset(X, Y, sample_size, random_state=RANDOM_STATE):
    if sample_size is None or sample_size >= len(X):
        return X, Y

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        test_size=len(X) - sample_size,
        random_state=random_state,
    )
    row_ids = np.arange(len(Y)).reshape(-1, 1)
    sample_index, _ = next(splitter.split(row_ids, Y))

    return [X[i] for i in sample_index], Y[sample_index]

def main():
    data = load_precomputed("./spectral_data/precomputed.pkl")
  
    X, Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y)
    X, Y = limit_dataset(X, Y, SAMPLE_SIZE)
    print(f"Using {len(X)} samples")
    X_train_val, Y_train_val, X_final_test, Y_final_test = train_test_split(X, Y)
    
    k_fold = MultilabelStratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_spec = None
    best_target_length = None
    best_validation_score = -math.inf

    for model_spec in MODELS:
        print(f"Evaluating {model_spec['name']}")
        X_model_train_val, target_length = prepare_model_input(
            X_train_val,
            model_spec["input_type"],
        )

        pairs = []

        for train_index, val_index in k_fold.split(X_model_train_val, Y_train_val):
            print("TRAIN:", train_index, "VALIDATION:", val_index)
            X_train, X_val = X_model_train_val[train_index], X_model_train_val[val_index]
            y_train, y_val = Y_train_val[train_index], Y_train_val[val_index]

            pairs.append((X_train, y_train, X_val, y_val))

        _best_model, validation_score = train_and_pick_best(model_spec["factory"], pairs)

        if validation_score > best_validation_score:
            print(f"Best model type so far: {model_spec['name']} ({validation_score} > {best_validation_score})")
            best_spec = model_spec
            best_target_length = target_length
            best_validation_score = validation_score

    print(f"Selected model type: {best_spec['name']} with validation F-beta score: {best_validation_score}")

    X_best_train_val, _ = prepare_model_input(
        X_train_val,
        best_spec["input_type"],
        target_length=best_target_length,
    )
    X_final_train, Y_final_train, X_threshold_val, Y_threshold_val = train_test_split(
        X_best_train_val,
        Y_train_val,
        ratio=0.875,
        random_state=7,
    )
    final_model = train_final_model(best_spec["factory"], X_final_train, Y_final_train, X_threshold_val, Y_threshold_val)

    X_final_test, _ = prepare_model_input(
        X_final_test,
        best_spec["input_type"],
        target_length=best_target_length,
    )
    final_test_loss = final_model.test(X_final_test, Y_final_test)
    print(f"Final holdout test loss: {final_test_loss}")
    final_test_score = score_multilabel_model(final_model, X_final_test, Y_final_test)
    print(f"Final holdout F-beta score: {final_test_score}")

    final_model.label_classes = mlb.classes_
    final_model.selected_model_type = best_spec["name"]
    final_model.input_type = best_spec["input_type"]
    final_model.target_length = best_target_length
    final_model.save(BEST_MODEL_PATH)
    

def test():
    data = load_precomputed("./spectral_data/precomputed.pkl")

    model: SVMModel = None
    with open(BEST_MODEL_PATH, "rb") as f:
        model = pkl.load(f)

    X,Y = convert_data_to_vectors(data)
    Y, mlb = process_labels(Y, classes=getattr(model, "label_classes", None))
    input_type = getattr(model, "input_type", "spectrogram")
    target_length = getattr(model, "target_length", None)

    if target_length is None and hasattr(model.model, "input_shape"):
        target_length = model.model.input_shape[1]

    X, _ = prepare_model_input(X, input_type, target_length=target_length)
    X, Y, X_test, Y_test = train_test_split(X, Y)
        
    Y_pred = model.predict(X_test)

    plot_multilabel_evaluation(
        y_true=Y_test,
        y_pred=Y_pred,
        label_names=[str(c) for c in mlb.classes_],
        save=True,
        output_prefix='plots/test_multilabel_eval'
    )




if __name__ == "__main__":
    #precompute()
    #main()
    test()
