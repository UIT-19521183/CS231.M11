import json
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
import joblib
import numpy as np
import pickle
from necessary_functions_for_train_model import *

def main():
    X_train_file='X_train5.csv'
    y_train_file='y_train5.csv'
    X_,y_=load_data(X_train_file,y_train_file)
    clf1=joblib.load('clf.pkl')
    scaler= joblib.load('scaler.save')
    training(X_,y_,clf1,scaler)

main()