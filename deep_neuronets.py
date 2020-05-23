# encode: utf-8

from prepare import *
from util_funcs import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.neural_network import MLPRegressor
from sklearn.tree import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

import pickle

import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasRegressor

import itertools
from datetime import datetime
import os

splitted_data, poly_splitted_data, norm_splitted_data = adv_split_data(get_prepared_data())

# один из вариантов моделей
# model = keras.models.Sequential([
#     keras.layers.InputLayer(X_train.shape[1]),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(100, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(25, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(20, activation='relu'),
#     keras.layers.Dense(1)
# ])

model = keras.models.Sequential([
    keras.layers.InputLayer(X_train.shape[1]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(90, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(45, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(40, activation='sigmoid'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
                        loss='mean_absolute_error',
                        metrics=['mae'])

model.fit(X_train, y_train, batch_size=70, epochs=40, callbacks=[
    keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=0, mode="min"),
])

print("Оценка:", mean_absolute_error(model.predict(X_test).astype(int), y_test))

save_keras_model_to_file(model)

def get_model(input_shape):
    def model_layer(X):
        X = keras.layers.Dense(20, activation='relu')(X)
        X = keras.layers.Dropout(0.2)(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dense(35, activation='relu')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dense(15, activation='relu')(X)
        return X
    
    def concat_layer(x):
        x1 = model_layer(x)
        x2 = keras.layers.Dense(12, activation='relu')(x)
        x = keras.layers.concatenate([x1, x2])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        return x
    
    input_model = keras.layers.Input(input_shape)
    x = input_model
    x = keras.layers.BatchNormalization()(input_model)
    x = concat_layer(x)
    x = keras.layers.Dense(40, activation='sigmoid')(x)
    predictions = keras.layers.Dense(1)(x)

    return keras.models.Model(inputs=input_model, outputs=predictions)

for i in range(50):
    model = get_model(X_train.shape[1])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    model.fit(X_train, y_train, batch_size=100, epochs=300, callbacks=[
        keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=0, mode="min"),
    ])
    print(mean_absolute_error(list(map(lambda x:x[0], model.predict(X_test).astype(int))), y_test))
    save_keras_model_to_file(model)