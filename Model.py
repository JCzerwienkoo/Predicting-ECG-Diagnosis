from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import pickle as pkl
import math
import os

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import keras
import keras_hub
from sklearn.metrics import fbeta_score
import tensorflow as tf

def tune_thresholds_per_label(y_true, y_prob, beta=2.0, grid=None, base_thresholds = None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    n_labels = y_true.shape[1]
    
    thresholds = np.full(n_labels, 0.5, dtype=float) if base_thresholds is None else base_thresholds

    for j in range(n_labels):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        best_t = 0.5
        best_s = -1.0

        for t in grid:
            yhat = (yp >= t).astype(int)
            s = fbeta_score(yt, yhat, beta=beta, zero_division=0)
            if s > best_s:
                best_s = s
                best_t = t

        thresholds[j] = best_t

    return thresholds

def score_multilabel_model(model, X, Y, beta=3.0):
    Y_pred = model.predict(X)
    return fbeta_score(Y, Y_pred, beta=beta, average="macro", zero_division=0)

def train_and_pick_best(get_model: Callable[[], 'Model'], train_test_pairs, model_params=()):
    best_score = -math.inf
    best_model = None
    
    i = 1
    
    for X_train, Y_train, X_test, Y_test in train_test_pairs:
        model = get_model(X_train.shape[1:], Y_train.shape[1:][0])
        print(f"Training {model.model_name} - {model.variant_name} - fold: {i}")
        
        model.train(X_train, Y_train)
        if hasattr(model, "tune_thresholds"):
            model.tune_thresholds(X_test, Y_test)
        
        score = score_multilabel_model(model, X_test, Y_test)
        
        if score > best_score:
            print(f"Trained model is superior {score} > {best_score}")
            best_score = score
            best_model = model
            
        i += 1
            
        
    return best_model, best_score

def train_final_model(get_model: Callable[[], 'Model'], X_train, Y_train, X_validation, Y_validation):
    model = get_model(X_train.shape[1:], Y_train.shape[1:][0])
    print(f"Training final {model.model_name} - {model.variant_name}")
    model.train(X_train, Y_train)

    if hasattr(model, "tune_thresholds"):
        model.tune_thresholds(X_validation, Y_validation)

    return model

class Model(metaclass=ABCMeta):
    def __init__(self, model_name, epoch_count):
        self.model_name = model_name
        self.variant_name = ""
        self.loss = math.inf
        self.model = None
        self.epoch_count = epoch_count
        
        self.training_history = []
    
    def save(self, output_path=None):
        os.makedirs("./models", exist_ok=True)
        if output_path is None:
            output_path = f"./models/{self.model_name}-{self.variant_name}.pkl"

        with open(output_path, "wb") as f:
            pkl.dump(self, f)
    
    @abstractmethod
    def train(self, X, Y) -> list[float]:
        pass
    
    @abstractmethod
    def _test(self, X, Y) -> float:
        pass
    
    def test(self, X, Y):
        results = self._test(X, Y)
        self.loss = results
        return results
    
    @abstractmethod
    def predict(self, X):
        pass
    
    
class SVMModel(Model):
    def __init__(self, variant_name, *kwargs, config={}):
        super().__init__("svm", config.get("epoch_count", 1))
        self.variant_name = variant_name
        self.model = MultiOutputClassifier(LinearSVC(*kwargs), n_jobs=14)
        
    def train(self, X: np.array, Y):
        self.model.fit(X, Y)
    
    def _test(self, X, Y):
        return 1 - self.model.score(X, Y)
    
    def predict(self, X):
        return self.model.predict(X)


class CNNModel(Model):
    def __init__(self, variant_name, input_shape, num_labels, config: dict = {}):
        super().__init__("cnn", config.get("epoch_count", 5))
        self.variant_name = variant_name
        self.model = self.create_model(input_shape, num_labels, config or {})
        
        self.thresholds = np.full(num_labels, 0.5, dtype=float)
        
        
    def create_model(self, input_shape, num_labels, config):
        inputs = keras.Input(shape=input_shape)
        
        hidden_layer_activation = config.get("hidden_layer_activation", "relu")
        kernel_size = config.get("kernel_size", (3,3))
        
        x = keras.layers.BatchNormalization()(inputs)
        
        x = keras.layers.Conv2D(64, kernel_size=kernel_size, padding="same", activation=hidden_layer_activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.15)(x)

        x = keras.layers.Conv2D(128, kernel_size=kernel_size, padding="same", activation=hidden_layer_activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.35)(x)

        x = keras.layers.Conv2D(256, kernel_size=kernel_size, padding="same", activation=hidden_layer_activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.35)(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.40)(x)

        outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_focal_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(threshold=0.5),
                keras.metrics.AUC(multi_label=True, num_labels=num_labels),
            ],
        )
        return model
        
        
    def train(self, X, Y):
        fit_results = self.model.fit(X, Y, epochs=self.epoch_count)
        return fit_results

    def tune_thresholds(self, X, Y):
        Y_pred = self.model.predict(X)
        self.thresholds = tune_thresholds_per_label(Y, Y_pred, beta=3.0, base_thresholds=self.thresholds)
    
    def _test(self, X, Y):
        return self.model.evaluate(X, Y, return_dict=True)["loss"]
    
    def predict(self, X):
        prediction = self.model.predict(X)
        
        y_pred = []
        for y in prediction:
            binarized = []
            for i in range(len(self.thresholds)):
                binarized.append(y[i] >= self.thresholds[i])
            y_pred.append(np.array(binarized).astype(int))

        return np.array(y_pred)
        

class ResNet(Model):
    def __init__(self, variant_name, input_shape, num_labels, config: dict = {}):
        super().__init__("resnet", config.get("epoch_count", 3))
        self.variant_name = variant_name
        self.model = self.create_model(input_shape, num_labels, config or {})
        
        self.thresholds = np.full(num_labels, 0.5, dtype=float)
        
        
    def create_model(self, input_shape, num_labels, config):
        inputs = keras.Input(shape=input_shape)
        
        hidden_layer_Activation = config.get("hidden_layer_activation", "relu")
        
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.Conv2D(64, kernel_size=(7, 7), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation=hidden_layer_Activation, kernel_initializer=keras.initializers.he_normal())(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_focal_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(threshold=0.5),
                keras.metrics.AUC(multi_label=True, num_labels=num_labels),
            ],
        )
        
        return model
        
        
    def train(self, X, Y):
        fit_results = self.model.fit(x=X, y=Y, epochs=self.epoch_count)
        return fit_results

    def tune_thresholds(self, X, Y):
        Y_pred = self.model.predict(X)
        self.thresholds = tune_thresholds_per_label(Y, Y_pred, beta=3.0, base_thresholds=self.thresholds)
    
    def _test(self, X, Y):
        return self.model.evaluate(X, Y, return_dict=True)["loss"]
    
    def predict(self, X):
        prediction = self.model.predict(X)
        
        y_pred = []
        for y in prediction:
            binarized = []
            for i in range(len(self.thresholds)):
                binarized.append(y[i] >= self.thresholds[i])
            y_pred.append(np.array(binarized).astype(int))

        return np.array(y_pred)
