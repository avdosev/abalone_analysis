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

params = [
{'base_estimator': MLPRegressor(hidden_layer_sizes=(5,), solver='lbfgs', random_state=42), 'n_estimators':x, 'random_state':42} for x in range(10,110,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42), 'n_estimators':x, 'random_state':42} for x in range(10,110,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(15,), solver='lbfgs', random_state=42), 'n_estimators':x, 'random_state':42} for x in range(10,60,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(20,), solver='lbfgs', random_state=42), 'n_estimators':x, 'random_state':42} for x in range(10,60,10)]
temp_errors = test_model_F(*splitted_data, BaggingRegressor, params, sep="\n\n", returnErrors = True)


params = list(itertools.chain.from_iterable(
[[
    {'base_estimator': MLPRegressor(hidden_layer_sizes=(x,y), solver='lbfgs', random_state=42), 'n_estimators':30, 'random_state':42}
    for x in (3,5,7,10)]
for y in (3,5,7,10)]))
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n\n")


params = [
{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', activation=x, random_state=42),
 "n_estimators":30, "random_state":42} for x in ('logistic', 'identity', 'tanh', 'relu')]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False)


params = [{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42), "n_estimators":30, "random_state":42}]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False)
test_model_F(*poly_splitted_data, BaggingRegressor, params, show_params=False)
test_model_F(*norm_splitted_data, BaggingRegressor, params, show_params=False)


params = [{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42),
           "n_estimators":30, "random_state":42, "max_samples":x} for x in (0.1,0.25,0.5,0.75,1.0)]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False)


params = [{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42),
           "n_estimators":30, "random_state":42, "max_features":x} for x in (0.1,0.25,0.5,0.75,1.0)]
test_model_F(*splitted_data, BaggingRegressor, params, show_params=False)