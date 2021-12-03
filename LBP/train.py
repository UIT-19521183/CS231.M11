import pandas as pd
from sklearn.svm import LinearSVC
import argparse
from Read_data_train import *


parser = argparse.ArgumentParser(description='Parse Detection Directory')
parser.add_argument('--input', help='Path to directory containing images for input')
parser.add_argument('--output', help='Path to directory for output')
parser.add_argument('--hinput', help='Path to directory for hard input')
parser.add_argument('--houtput', help='Path to directory for hard output')

args = parser.parse_args()
X = pd.read_csv(args.input)
y = pd.read_csv(args.output)

X_new = X.iloc[:, 1:]
y_new = y.iloc[:, 1:]

#clf1 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
#clf1.fit(X.values, y) 
#joblib.dump(clf1, 'pedestrian.pkl')

#tạo scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_new)
X_new=scaler.transform(X_new)
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
clf1.fit(X_new, y) 
#Quyết định dùng SVM với kernel=rbf và cell_size=8x8 và C=0.5
joblib.dump(clf1, 'pedestrian_final_1.pkl')

# Trích xuất thêm đặc trưng của các ảnh không chứa người khó, sau đó hợp với các đặc trưng cũ và train lại
hard_neg_limit = 10000
X_hard, y_hard, hard_neg_count = read_hard_negative_images(neg_dir, hard_neg_limit, clf1)

X_hard = np.concatenate((X_hard, X), axis = 0)
y_hard = np.concatenate((y_hard, y), axis = 0)

print('Continue training hard negative images')
print('Number of negative images:', hard_neg_count)

X_hard = pd.read_csv(args.hinput)
y_hard = pd.read_csv(args.houtput)

X_hard_new = X_hard.iloc[:, 1:]
y_hard_new = y_hard.iloc[:, 1:]

scaler = StandardScaler()
scaler.fit(X_hard_new)
X_hard_new=scaler.transform(X_hard_new)
scaler_hard_filename = "scaler_hard.save"
joblib.dump(scaler, scaler_hard_filename)

#Train thử, quyết định sẽ dùng classifier nào để train data, loại bỏ False Positive
clf = SVC(C=10000, max_iter=30000, class_weight='balanced', verbose=1) #cell_size=8x8
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
clf.fit(X_hard_new, y_hard_new) 
#Quyết định dùng SVM với kernel=rbf và cell_size=8x8 và C=0.5
joblib.dump(clf, 'pedestrian_final.pkl')