from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import pickle as pkl
import math

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import keras
from sklearn.metrics import fbeta_score
import tensorflow as tf

from processing import flatten_data_linear, transpose_entries

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

def train_and_pick_best(get_model: Callable[[], 'Model'], train_test_pairs, model_params=()):
    best_loss = math.inf
    best_model = None
    
    i = 1
    
    for X_train, Y_train, X_test, Y_test in train_test_pairs:
        model = get_model(X_train.shape[1:], Y_train.shape[1:][0])
        print(f"Training {model.model_name} - {model.variant_name} - fold: {i}")
        
        model.train(X_train, Y_train)
        
        loss = model.test(X_test, Y_test)
        
        if loss < best_loss:
            print(f"Trained model is superior {loss} < {best_loss}")
            best_loss = loss
            best_model = model
            
        i += 1
            
        
    return best_model, best_loss

class Model(metaclass=ABCMeta):
    def __init__(self, model_name, epoch_count):
        self.model_name = model_name
        self.variant_name = ""
        self.loss = math.inf
        self.model = None
        self.epoch_count = epoch_count
        
        self.training_history = None
    
    def save(self):
        with open(f"./models/{self.model_name}-{self.variant_name}.pkl", "wb") as f:
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
    
    @abstractmethod
    def transform_data(self, X):
        pass
    
    
class SVMModel(Model):
    def __init__(self, variant_name, *kwargs, config = {}):
        super().__init__("svm", config.get("epoch_count", 1))
        self.variant_name = variant_name
        self.model = MultiOutputClassifier(LinearSVC(*kwargs), n_jobs=14)
        
    def train(self, X: np.array, Y):
        X = self.transform_data(X)
        X = X[:, :, 0]
        self.model.fit(X, Y)
    
    def _test(self, X, Y):
        X = self.transform_data(X)
        X = X[:, :, 0]
        return 1 - self.model.score(X, Y)
    
    def predict(self, X):
        X = self.transform_data(X)
        X = X[:, :, 0]
        return self.model.predict(X)
    
    def transform_data(self, X):
        return flatten_data_linear(X)


class CNNModel(Model):
    def __init__(self, variant_name, input_shape, num_labels, config: dict = {}):
        super().__init__("cnn", config.get("epoch_count", 20))
        self.variant_name = variant_name
        # hackish
        transformed_shape = self.transform_data(np.zeros([1, *input_shape])).shape[1:]
        self.model = self.create_model(transformed_shape, num_labels, config or {})
        
        self.thresholds = np.full(num_labels, 0.5, dtype=float)
        
        
    def create_model(self, input_shape, num_labels, config):
        inputs = keras.Input(shape=input_shape)
       
        layers = config.get("layers")
        
        x = keras.layers.BatchNormalization()(inputs)
        for layer in layers:
            x = layer(x)
            
        
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
        X = self.transform_data(X)
        fit_results = self.model.fit(X, Y, epochs=self.epoch_count)
        
        self.training_history = fit_results
        
        # self.thresholds = tune_thresholds_per_label(Y, Y_pred, base_thresholds=self.thresholds)
        
        return fit_results
    
    def _test(self, X, Y):
        X = self.transform_data(X)
        return self.model.evaluate(X, Y, return_dict=True)["loss"]
    
    def predict(self, X):
        X = self.transform_data(X)
        prediction = self.model.predict(X)
        
        y_pred = []
        for y in prediction:
            binarized = []
            for i in range(len(self.thresholds)):
                binarized.append(y[i] >= self.thresholds[i])
            y_pred.append(np.array(binarized).astype(int))

        return np.array(y_pred)
    
    def transform_data(self, X):
        return transpose_entries(X)
        

class ResNet(Model):
    def __init__(self, variant_name, input_shape, num_labels, config: dict = {}):
        super().__init__("resnet", config.get("epoch_count", 3))
        self.variant_name = variant_name
        # hackish
        transformed_shape = self.transform_data(np.zeros([1, *input_shape])).shape[1:]
        self.model = self.create_model(transformed_shape, num_labels, config or {})
        
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
        outputs = keras.layers.Dense(num_labels, activation="softmax")(x)

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
        X = self.transform_data(X)
        fit_results = self.model.fit(x=X, y=Y, epochs=self.epoch_count)
        
        self.training_history = fit_results
        
        # Y_pred = self.model.predict(X)
        
        # self.thresholds = tune_thresholds_per_label(Y, Y_pred, base_thresholds=self.thresholds)
        
        return fit_results
    
    def _test(self, X, Y):
        X = self.transform_data(X)
        return self.model.evaluate(X, Y, return_dict=True)["loss"]
    
    def predict(self, X):
        X = self.transform_data(X)
        prediction = self.model.predict(X)
        
        y_pred = []
        for y in prediction:
            binarized = []
            for i in range(len(self.thresholds)):
                binarized.append(y[i] >= self.thresholds[i])
            y_pred.append(np.array(binarized).astype(int))

        return np.array(y_pred)
    
    def transform_data(self, X):
        return transpose_entries(X)