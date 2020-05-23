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

test_model_F(*splitted_data, LinearRegression, saveModel=True)

test_model_F(*poly_splitted_data, LinearRegression, saveModel=True)

params=[
    {'kernel': 'linear', 'gamma':'auto'}, 
    {'kernel': 'poly', 'degree': 2, 'gamma':'auto'}, 
    {'kernel': 'rbf', 'gamma':'auto'}, 
    {'kernel': 'sigmoid', 'gamma':'auto'}]
test_model_F(*splitted_data, SVC, params, saveModel=True)

test_model_F(*splitted_data, SVR, params=[
    {'kernel': 'linear', 'gamma':'auto'}, 
    {'kernel': 'poly', 'degree': 2, 'gamma':'auto'}, 
    {'kernel': 'rbf', 'gamma':'auto'}], saveModel=True)

params = [
    {'C':1.0, 'max_iter':10000}, 
    {'C':5.0, 'max_iter':100000}, 
    {'C':125.0, 'max_iter':100000},
    {'C':625.0, 'max_iter':100000},
    {'C':2500.0, 'max_iter':100000},
    {'C':12500.0, 'max_iter':100000},
    {'C':62500.0, 'max_iter':100000},]
test_model_F(*splitted_data, LinearSVR, params=params, saveModel=True)

params = [{'nu':x, 'gamma':'auto'} for x in [.01,.1,.2,.3,.4,.5,.6,.8,1.0]]
test_model_F(*splitted_data, NuSVR, params=params, saveModel=True)

params = [{"max_depth":x} for x in range(1,10)]
test_model_F(*splitted_data, DecisionTreeRegressor, params, saveModel=True)

params = [{'n_neighbors':x} for x in (3,4,5,6,7,8,9,10)]
test_model_F(*splitted_data, KNeighborsRegressor, params = params, saveModel=True)

params = [{'alpha':x} for x in (0.001, 0.01, 0.1, 1.0, 10.0)]
test_model_F(*splitted_data, KernelRidge, params = params, saveModel=True)