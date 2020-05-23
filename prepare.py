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

import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasRegressor

import itertools
from datetime import datetime
import os

def get_prepared_data():
    data_path = './data/abalone.csv'
    df = pd.read_csv(data_path)
    df.Sex = df.Sex.transform(lambda x:{"M": 0.5,"F": 1,"I": 0}[x])
    df.drop(['Diameter', 'Height'], axis=1, inplace=True)
    return df

def split_data(df):
    X_data, Y_data = df.loc[:,:"Shell"], df["Rings"]
    X_train,  X_test,  y_train,  y_test =  train_test_split(X_data, Y_data,  test_size= 0.20,  random_state= 42 )
    splitted_data = [X_train, y_train, X_test, y_test]
    return splitted_data

def adv_split_data(df):
    X_data, Y_data = df.loc[:,:"Shell"], df["Rings"]
    X_train,  X_test,  y_train,  y_test =  train_test_split(X_data, Y_data,  test_size= 0.20,  random_state= 42 )
    splitted_data = [X_train, y_train, X_test, y_test]
    # данные, дополненные полиномами
    poly = PolynomialFeatures(degree = 2)
    poly_splitted_data = [poly.fit_transform(X_train), y_train, poly.transform(X_test), y_test]
    # нормализованные данные
    scaler = StandardScaler()
    scaler.fit(X_train)
    norm_X_train = scaler.transform(X_train)
    norm_X_test = scaler.transform(X_test)
    norm_splitted_data = [norm_X_train, y_train, norm_X_test, y_test]
    return splitted_data, poly_splitted_data, norm_splitted_data