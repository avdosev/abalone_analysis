# encode: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pickle

from datetime import datetime
import os
from time import time

def test_model_F(train_X, train_y, test_X, test_y, model, params=[{}], sep='',
 show_params=True, return_errors=False, print_errors=True, save_model=True, folder_save_to = './test_models/', preprint_message = '', check_time = False):
    if return_errors:
        res = []
    for param in params:
        test_model = model(**param)
        if check_time:
            time_start = time()
        test_model.fit(train_X, train_y)
        if check_time:
            time_end = time()
        if preprint_message!='':
            print(preprint_message)
        if params != [{}] and show_params:
            print("Параметры:", param)
        if print_errors:
            print("Оценка:", mean_absolute_error(test_model.predict(test_X).astype(int), test_y))
            print(end=sep)
        if check_time:
            print(f"Время: {time_end-time_start}")
        if return_errors:
            res.append(mean_absolute_error(test_model.predict(test_X).astype(int), test_y))
        if save_model:
            save_to_file(test_model, folder_save_to)
    if return_errors:
        return res
    
def calc_errors(predicted_y, real_y):
    y_diff = list(map(lambda x,y:x-y, predicted_y ,real_y))
    me = sum(y_diff)/len(y_diff)
    mae = mean_absolute_error(predicted_y, real_y)
    mse = mean_squared_error(predicted_y, real_y)
    print(f"Средняя ошибка: {me}")
    print(f"Средняя абсолютная ошибка: {mae}")
    print(f"Средняя квадратичная ошибка: {mse}")
    print("Распределение ошибок:")
    plt.figure()
    plt.hist(y_diff, bins=int(max(y_diff)-min(y_diff)+1))
    plt.show()
    
def int_mae(y_true, y_pred):
    return np.mean(list(map(lambda x,y: int(abs(x-y)), y_true,y_pred)))

def int_mse(y_true, y_pred):
    return np.mean(list(map(lambda x,y: int(x-y)**2, y_true,y_pred)))

def get_name_for_model(folder = './test_models/'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    cur_time = datetime.now().isoformat().replace(":","_").replace(".","_")
    filename = folder+'model_'+cur_time
    return filename

def save_to_file(model, folder = './test_models/'):
    filename = get_name_for_model(folder)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Модель сохранена в " + filename)
    
def save_keras_model_to_file(model):
    name = get_name_for_model('./test/keras_models/')
    os.mkdir(name)
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