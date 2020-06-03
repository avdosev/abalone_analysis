# encode: utf-8

from prepare import *
from util_funcs import *

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

from datetime import datetime
import os

folder_save_to = './test/simple_models/'

splitted_data, poly_splitted_data, norm_splitted_data = adv_split_data(get_prepared_data())

test_model_F(*splitted_data, LinearRegression, preprint_message = 'LinearRegression', folder_save_to = folder_save_to)

test_model_F(*poly_splitted_data, LinearRegression, preprint_message = 'polynomial LinearRegression', folder_save_to = folder_save_to)

params=[
    {'kernel': 'linear', 'gamma':'auto'}, 
    {'kernel': 'poly', 'degree': 2, 'gamma':'auto'}, 
    {'kernel': 'rbf', 'gamma':'auto'}, 
    {'kernel': 'sigmoid', 'gamma':'auto'}]
test_model_F(*splitted_data, SVC, params, preprint_message = 'SVC', folder_save_to = folder_save_to)

test_model_F(*splitted_data, SVR, params=[
    {'kernel': 'linear', 'gamma':'auto'}, 
    {'kernel': 'poly', 'degree': 2, 'gamma':'auto'}, 
    {'kernel': 'rbf', 'gamma':'auto'}], preprint_message = 'SVR', folder_save_to = folder_save_to)

params = [
    {'C':1.0, 'max_iter':10000}, 
    {'C':5.0, 'max_iter':100000}, 
    {'C':125.0, 'max_iter':100000},
    {'C':625.0, 'max_iter':100000},
    {'C':2500.0, 'max_iter':100000},
    {'C':12500.0, 'max_iter':100000},
    {'C':62500.0, 'max_iter':100000},]
test_model_F(*splitted_data, LinearSVR, params=params, preprint_message = 'LinearSVR', folder_save_to = folder_save_to)

params = [{'nu':x, 'gamma':'auto'} for x in [.01,.1,.2,.3,.4,.5,.6,.8,1.0]]
test_model_F(*splitted_data, NuSVR, params=params, preprint_message = 'NuSVR', folder_save_to = folder_save_to)

params = [{"max_depth":x} for x in range(1,10)]
test_model_F(*splitted_data, DecisionTreeRegressor, params, preprint_message = 'DecisionTreeRegressor', folder_save_to = folder_save_to)

params = [{'n_neighbors':x} for x in (3,4,5,6,7,8,9,10)]
test_model_F(*splitted_data, KNeighborsRegressor, params = params, preprint_message = 'KNeighborsRegressor', folder_save_to = folder_save_to)

params = [{'alpha':x} for x in (0.001, 0.01, 0.1, 1.0, 10.0)]
test_model_F(*splitted_data, KernelRidge, params = params, preprint_message = 'KernelRidge', folder_save_to = folder_save_to)