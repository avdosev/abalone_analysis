# encode: utf-8

from prepare import *
from util_funcs import *

from sklearn.ensemble import *
from sklearn.neural_network import MLPRegressor

import itertools
import os

folder_save_to = './test/neuro_ensembles/'

splitted_data, poly_splitted_data, norm_splitted_data = adv_split_data(get_prepared_data())

print("\nтест функции оптимизации\n")
params = list(itertools.chain.from_iterable(
[[
    {'base_estimator': MLPRegressor(hidden_layer_sizes=(n,), solver=s, random_state=42, max_iter=5000), 'n_estimators':30, 'random_state':42}
    for n in (3,5,7,10,15,20)]
for s in ('lbfgs','sgd', 'adam')]))
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", check_time = True, folder_save_to = folder_save_to)

print("\nтест количества нейронов в однослойной сети\n")
params = [
{'base_estimator': MLPRegressor(hidden_layer_sizes=(5,), solver='lbfgs', random_state=42, max_iter=5000), 'n_estimators':x, 'random_state':42} for x in range(10,110,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42, max_iter=5000), 'n_estimators':x, 'random_state':42} for x in range(10,110,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(15,), solver='lbfgs', random_state=42, max_iter=5000), 'n_estimators':x, 'random_state':42} for x in range(10,60,10)]+[
{'base_estimator': MLPRegressor(hidden_layer_sizes=(20,), solver='lbfgs', random_state=42, max_iter=5000), 'n_estimators':x, 'random_state':42} for x in range(10,60,10)]
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", folder_save_to = folder_save_to)

print("\nтест количества нейронов в двуслойной сети\n")
params = list(itertools.chain.from_iterable(
[[
    {'base_estimator': MLPRegressor(hidden_layer_sizes=(x,y), solver='lbfgs', random_state=42, max_iter=5000), 'n_estimators':30, 'random_state':42}
    for x in (3,5,7,10,15,20)]
for y in (3,5,7,10,15,20)]))
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", folder_save_to = folder_save_to)

print("\nтест разных функций активации\n")
params = list(itertools.chain.from_iterable(
[[
    {'base_estimator': MLPRegressor(hidden_layer_sizes=(n,), solver='lbfgs', activation=x, random_state=42, max_iter=5000),
    "n_estimators":30, "random_state":42} 
for x in ('logistic', 'identity', 'tanh', 'relu')]
for n in (5,10,15,20)]))
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", folder_save_to = folder_save_to)

print("\nтест max_samples\n")
params = [{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42, max_iter=5000),
           "n_estimators":30, "random_state":42, "max_samples":x} for x in (0.1,0.25,0.5,0.75,1.0)]
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", folder_save_to = folder_save_to)

print("\nтест max_features\n")
params = [{'base_estimator': MLPRegressor(hidden_layer_sizes=(10,), solver='lbfgs', random_state=42, max_iter=5000),
           "n_estimators":30, "random_state":42, "max_features":x} for x in (0.1,0.25,0.5,0.75,1.0)]
test_model_F(*splitted_data, BaggingRegressor, params, sep="\n", folder_save_to = folder_save_to)