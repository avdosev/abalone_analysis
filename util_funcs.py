# encode: utf-8

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

import itertools
from datetime import datetime
import os

def test_model_F(train_X, train_y, test_X, test_y, model, params=[{}], sep='',
                 show_params=True, returnErrors=False, printErrors=True, saveModel=False):
    if returnErrors:
        res = []
    for param in params:
        test_model = model(**param)
        test_model.fit(train_X, train_y)
        if params != [{}] and show_params:
            print("Параметры:", param)
        if printErrors:
            print("Оценка:", mean_absolute_error(test_model.predict(test_X).astype(int), test_y))
            print(end=sep)
        if returnErrors:
            res.append(mean_absolute_error(test_model.predict(test_X).astype(int), test_y))
        if saveModel:
            save_to_file(test_model)
            
    if returnErrors:
        return res
    
def calc_errors(predicted_y, real_y):
    y_diff = list(map(lambda x,y:x-y, predicted_y ,real_y))
    me = sum(y_diff)/len(y_diff)
    mae = mean_absolute_error(predicted_y, real_y)
    print(f"Средняя ошибка: {me}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print("Распределение ошибок:")
    fig = plt.figure()
    plt.hist(y_diff, bins=max(y_diff)-min(y_diff)+1)
    plt.show()
    
def int_mae(y_true, y_pred):
    return np.mean(list(map(lambda x,y: int(abs(x-y)), y_true,y_pred)))

def int_mse(y_true, y_pred):
    return np.mean(list(map(lambda x,y: int(x-y)**2, y_true,y_pred)))

def get_name_for_model():
    cur_time = datetime.now().isoformat().replace(":","_").replace(".","_")
    filename = './test_models/'+'model_'+cur_time+'.hdf5'
    return filename

def save_to_file(model):
    filename = get_name_for_model()
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Модель сохранена в " + filename)
    
def save_keras_model_to_file(model):
    name = get_name_for_model()
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    model.save(name)
    print("Модель сохранена в " + name)

def load_from_file(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def showG(x_data, model):
    data_predicted = x_data.copy()
    Y_predicted = model.predict(data_predicted)
    data_predicted['rings'] = Y_predicted.astype(int)
    sns.set()
    sns.pairplot(data_predicted, height = 6)
    plt.show()