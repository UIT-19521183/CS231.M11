import pandas as pd
from sklearn.svm import LinearSVC
import argparse
from Read_data_train import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X1 = pd.read_csv('X_train_new.csv')
y1 = pd.read_csv('y_train_new.csv')

X2 = X1.iloc[:, 1:]
y2 = y1.iloc[:, 1:]

clf1 = SVC(C=10000, max_iter=30000, class_weight='balanced', verbose=1)
#clf1 = SVC(C=9.9, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf1 = SVC(C=1000, max_iter=100000, class_weight='balanced', verbose=1, probability=True) #cell_size=16x16
#clf1 = SVC(C=155, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=32x32
#clf1 = SVC(C=4.11, max_iter=51000, class_weight='balanced', verbose=1) #cell_size=24x24
#clf1 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
#clf1=LinearSVC(C=1, max_iter=10000, class_weight='balanced', verbose=1) # nhiều bouding box chồng chéo, không detect đúng
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
clf1.fit(X2.values, y2) 
joblib.dump(clf1, 'pedestrian_only.pkl')

X1_hard = pd.read_csv('X_train_hard_new.csv')
y1_hard = pd.read_csv('y_train_hard_new.csv')

X2_hard = X1_hard.iloc[:, 1:]
y2_hard = y1_hard.iloc[:, 1:]

clf2 = SVC(C=10000, max_iter=30000, class_weight='balanced', verbose=1)
#clf2 = SVC(C=9.9, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf2 = SVC(C=1000, max_iter=100000, class_weight='balanced', verbose=1, probability=True) #cell_size=16x16
#clf2 = SVC(C=155, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=32x32
#clf2 = SVC(C=4.11, max_iter=51000, class_weight='balanced', verbose=1) #cell_size=24x24
#clf2 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
#clf2 = LinearSVC(C=1, max_iter=10000, class_weight='balanced', verbose=1) #cell_size=8x8
#clf2 = LinearSVC(C=0.56, max_iter=1000, class_weight='balanced', verbose=1) #cell_size=8x8 #block=4x4
#clf2 = LinearSVC(C=1, max_iter=10000, class_weight='balanced', verbose=1) #cell_size=16x16
#clf2 = LinearSVC(C=1, max_iter=100000, class_weight='balanced', verbose=1) #cell_size=32x32
#clf2 = LinearSVC(C=0.11, max_iter=20000, class_weight='balanced', verbose=1) #cell_size=24x24
#clf2 = LogisticRegression(max_iter=10000,class_weight=None,C=0.0009) #cell_size=8x8
#clf2 = LogisticRegression(max_iter=10000,class_weight='balanced',C=5) #cell_size=8x8 block=4x4
#clf2 = LogisticRegression(max_iter=10000,class_weight=None,C=0.0285) #cell_size=16x16
#clf2 = LogisticRegression(max_iter=10000,class_weight=None,C=0.1) #cell_size=32x32
#clf2 = LogisticRegression(max_iter=10000,class_weight=None,C=0.025)  #cell_size=24x24
#clf2 = DecisionTreeClassifier(random_state=500,class_weight='balanced')

clf2.fit(X2_hard.values, y2_hard)
joblib.dump(clf2, 'pedestrian_final_only.pkl')
print('Training done')
