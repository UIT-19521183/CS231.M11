from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
import joblib

#Hàm load data
def load_data(X_train_file,y_train_file):
    #Load data
    X=pd.read_csv('X_train5.csv')
    y=pd.read_csv('y_train5.csv')
    #Tạo ra tên columns (do khi load data thì sẽ lấy hàng đầu làm columns nếu k có tên
    #Tạo tên = số từ 0->...
    columns_name_x=list(range(X.shape[1]))
    #y chỉ chứa 1 cột là kết quả
    columns_name_y=['0']
    X_=pd.read_csv('X_train5.csv',names=columns_name_x)
    y_=pd.read_csv('y_train5.csv',names=columns_name_y)
    X_=np.array(X_)
    y_=np.array(y_)
    y_=y_.flatten()
    return X_,y_

#Hàm lấy các sample True Positive
def get_FN_sample(X_test, y_test, pred):
    X_FN=[]
    y_FN=[]
    for i in range(len(X_test)):
        if(y_test[i]==1 and pred[i]==0):
            X_FN.append(X_test[i])
            y_FN.append(y_test[i])
    return X_FN,y_FN
def get_num_of_FP(X_test,y_test,pred):
    dem=0
    for i in range(len(X_test)):
        if (y_test[i] == 0 and pred[i] == 1):
            dem+=1
    return dem

#Hàm preidict + metric classifier
def predict_(X_test_file,y_test_file,clf1,scaler):
    if(str(type(X_test_file))=="<class 'str'>"):
        #Load test data
        X=pd.read_csv(X_test_file)
        y=pd.read_csv(y_test_file)
        #Tạo ra tên columns (do khi load data thì sẽ lấy hàng đầu làm columns nếu k có tên
        #Tạo tên = số từ 0->...
        columns_name_x=list(range(X.shape[1]))
        #y chỉ chứa 1 cột là kết quả
        columns_name_y=['0']
        X_test=pd.read_csv(X_test_file,names=columns_name_x)
        y_test=pd.read_csv(y_test_file,names=columns_name_y)
        X_test=np.array(X_test)
        y_test=np.array(y_test)
        y_test=y_test.flatten()

    else:
        X_test=X_test_file
        y_test=y_test_file

    # scale data
    X_test = scaler.transform(X_test)

    #Predict
    pred=clf1.predict(X_test)
    print("Accuracy score:",accuracy_score(y_test,pred))
    print("F1 score:",f1_score(y_test,pred))
    print("Precision score:",precision_score(y_test,pred))
    print("Recall score:",recall_score(y_test,pred))
    #print(np.max(X_train[0]),np.min(X_train[0]))
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    #plot_confusion_matrix(clf1, X_test, y_test)
    cm = confusion_matrix(y_test, pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative', 'Positive']);
    plt.show()
    X_FP,y_FP=get_FN_sample(X_test, y_test, pred)
    return X_FP,y_FP


from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler
#Hàm train bằng k-fold
def train_by_k_fold(X,y,n_split,classifier):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1)
    lst_accu_stratified = []
    lst_f1_stratified=[]
    lst_precision_stratified=[]
    lst_recall_stratified=[]
    num_of_FP=0
    num_of_FN=0
    X_FN=[]
    y_FN=[]
    for train_index, test_index in skf.split(X, y):
        #x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        #scale data
        scaler = StandardScaler()
        scaler.fit(x_train_fold)
        x_train_fold = scaler.transform(x_train_fold)
        x_test_fold=scaler.transform(x_test_fold)

        classifier.fit(x_train_fold, y_train_fold)
        y_pred=classifier.predict(x_test_fold)
        lst_accu_stratified.append(accuracy_score(y_test_fold,y_pred))
        lst_f1_stratified.append(f1_score(y_test_fold,y_pred,average='macro'))
        lst_precision_stratified.append(precision_score(y_test_fold,y_pred))
        lst_recall_stratified.append(recall_score(y_test_fold,y_pred))

        X_FN_temp,y_FN_temp=get_FN_sample(x_test_fold, y_test_fold, y_pred)
        X_FN=X_FN+X_FN_temp
        y_FN=y_FN+y_FN_temp
        num_of_FP+=get_num_of_FP(x_test_fold,y_test_fold,y_pred)
        num_of_FN+=len(X_FN_temp)
    #print('List of possible accuracy:', lst_accu_stratified)
    max_acc=max(lst_accu_stratified)
    min_acc=min(lst_accu_stratified)
    mean_acc=mean(lst_accu_stratified)
    max_f1=max(lst_f1_stratified)
    min_f1=min(lst_f1_stratified)
    mean_f1=mean(lst_f1_stratified)
    deviation_acc=stdev(lst_accu_stratified)
    deviation_f1=stdev(lst_f1_stratified)
    mean_pre=mean(lst_precision_stratified)
    mean_recall=mean(lst_recall_stratified)
    print('\nMaximum Accuracy That can be obtained from this model is:',
        max_acc*100, '%')
    print('\nMinimum Accuracy:',
        min_acc*100, '%')
    print('\nOverall Accuracy:',
        mean_acc*100, '%')
    print('\nStandard Deviation is:', deviation_acc)

    print('\nMaximum f1 That can be obtained from this model is:',
        max_f1*100, '%')
    print('\nMinimum f1:',
        min_f1*100, '%')
    print('\nOverall f1:',mean_f1*100, '%')
    print('\nStandard Deviation is:', deviation_f1)
    print('\nOverall Precision:',mean_pre*100, '%')
    print('\nOverall recall:',mean_recall*100, '%')
    print("\nNumber of False Positive:",num_of_FP)
    print("\nNumber of False Negative:",num_of_FN)
    return X_FN,y_FN

def training(X_,y_,clf1,scaler):
    #Tính accuracy và f1 khi train lần đầu trước khi HARD NEGATIVE-MINING
    from sklearn.model_selection import train_test_split
    X_test_file='X_test5.csv'
    y_test_file='y_test5.csv'
    #scale X_test,y_test
    X_test,y_test=load_data(X_test_file,y_test_file)
    X_test=scaler.transform(X_test)
    #train trước khi gộp X_FN
    clf1.fit(X_,y_)
    joblib.dump(clf1, 'SVM_rbf1.pkl')
    predict_(X_test,y_test,clf1,scaler)


    #X_FP,y_FP=train_by_k_fold(X_val,y_val,clf1,scaler2)
    #Lấy X_FP và y_FP
    X_FN,y_FN=train_by_k_fold(X_,y_,n_split=5,classifier=clf1)

    #Gộp train và FP vào để train lại
    X_train2=np.concatenate((X_, X_FN), axis=0)
    y_train2=np.concatenate((y_, y_FN), axis=0)

    #Train_data lần 2
    #clf1.fit(X_train2,y_train2)
    clf1.fit(X_train2,y_train2)
    joblib.dump(clf1, 'SVM_rbf2.pkl')

    #Measure accuracy và f1 = tập test
    predict_(X_test,y_test,clf1,scaler)

    print('Training done')
