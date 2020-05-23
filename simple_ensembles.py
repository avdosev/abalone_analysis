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

params = [{'random_state':42, "n_estimators":x} for x in (
    list(range(1,10,2)) + list(range(10,100,20)) + list(range(100,900,100)))
]
test_model_F(*splitted_data, GradientBoostingRegressor, params, saveModel=True)

params = [{'random_state':42, "n_estimators":x} for x in (
    list(range(1,10,2)) + list(range(10,100,20)) + list(range(100,500,100)))
]
test_model_F(*splitted_data, RandomForestRegressor, params, saveModel=True)

params = [{'random_state':42, "n_estimators":x} for x in (
    list(range(1,10,2)) + list(range(10,100,20)) + [100, 200])
]
test_model_F(*splitted_data, AdaBoostRegressor, params, saveModel=True)

m1 = GradientBoostingRegressor(random_state = 42, n_estimators=95)
m2 = RandomForestRegressor(random_state = 42, n_estimators=300)
m3 = LinearRegression()

params = [
    {'estimators':[('gb', m1), ('rf', m2), ('lr', m3)]},
    {'estimators':[('gb', m1), ('rf', m2)]},
    {'estimators':[('gb', m1), ('lr', m3)]},
    {'estimators':[('rf', m2), ('lr', m3)]}
]
test_model_F(*splitted_data, VotingRegressor, params, show_params=False, saveModel=True)

params = [
    {'base_estimator': DecisionTreeRegressor(), 'n_estimators':x, 'random_state':42} for x in (100,1000)
]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False, saveModel=True)


params = [
    {'base_estimator': LinearRegression(), 'n_estimators':x, 'random_state':42} for x in (10,100,1000)
]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False, saveModel=True)


params=[
    {'base_estimator': LinearRegression(), 'n_estimators':x, 'random_state':42} for x in (10,100,1000)
]
test_model_F(*poly_splitted_data, BaggingRegressor, params, show_params=False, saveModel=True)