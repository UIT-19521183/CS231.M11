import json
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
import joblib
import numpy as np
import pickle
from necessary_functions_for_train_model import *

X_train_file='X_train5.csv'
y_train_file='y_train5.csv'
X_,y_=load_data(X_train_file,y_train_file)

#tạo scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_)
X_=scaler.transform(X_)
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

#Ở đây chỉ có data train k cần chia

print('Training started')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#Train thử, quyết định sẽ dùng classifier nào để train data, loại bỏ False Positive
clf1 = SVC(C=10000, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8
#clf1 = SVC(C=9.9, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf1 = SVC(C=9.9, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf1 = SVC(C=1000, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=16x16
#clf1 = SVC(C=155, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=32x32
#clf1 = SVC(C=4.11, max_iter=51000, class_weight='balanced', verbose=1) #cell_size=24x24
#clf1=LinearSVC(C=1, max_iter=10000, class_weight='balanced', verbose=1) #cell_size=8x8
#clf1=LinearSVC(C=0.56, max_iter=1000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf1=LinearSVC(C=1, max_iter=10000, class_weight='balanced', verbose=1) #cell_size=16x16
#clf1=LinearSVC(C=1, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=32x32
#clf1=LinearSVC(C=0.11, max_iter=20000, class_weight='balanced', verbose=1) #cell_size=24x24
#clf1=LogisticRegression(max_iter=10000,class_weight=None,C=0.0009) #cell_size=8x8
#clf1=LogisticRegression(max_iter=10000,class_weight='balanced',C=5) #cell_size=8x8 block=4x4
#clf1=LogisticRegression(max_iter=10000,class_weight=None,C=0.0285) #cell_size=16x16
#clf1=LogisticRegression(max_iter=10000,class_weight=None,C=0.1) #cell_size=32x32
#clf1=LogisticRegression(max_iter=10000,class_weight=None,C=0.025)  #cell_size=24x24
#clf1=DecisionTreeClassifier(random_state=500,class_weight='balanced')
train_by_k_fold(X_,y_,n_split=5,classifier=clf1)
#Quyết định dùng SVM với kernel=rbf và cell_size=8x8 và C=0.5
joblib.dump(clf1, 'SVM_rbf_cell_size_8x8.pkl')
