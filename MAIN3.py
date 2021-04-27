'''
Created on Apr 15, 2021

@author: 14076
'''
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import time
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


def KNN(data,labels,X_train, X_test, Y_train, Y_test,features):
    if (features=='original'):
       print("\nUsing Original Features")
       X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size=0.2)

    if (features == 'reduced'):
        print("\nUsing Reduced Features")

    k_range = list(range(1, 30))
    weight_options = ["uniform", "distance"]
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
    grid.fit(X_train, Y_train)
    print('Best KNN Prameters are: ', grid.best_params_)
    knn = KNeighborsClassifier(**grid.best_params_)  # best paremetre olarak gelen değerlerimiz.
    start_time = time.time()
    knn.fit(X_train, Y_train)
    end_time = time.time()
    print("Time spent in Training=", end_time - start_time)
    scores = cross_val_score(knn, X_train,Y_train, cv=10)
    print("Average Cross validation accuracy=", np.sum(scores) / 10)
    y_pred_test = knn.predict(X_test)
    cm_test = confusion_matrix(Y_test, y_pred_test)
    acc_test = accuracy_score(Y_test, y_pred_test)
    print("Test Score: {} ".format(acc_test))
    print("Confusion Matrix Test: ", cm_test)

    disp = plot_confusion_matrix(knn, X_test,Y_test,
                                 display_labels=['Benign' ,'Malignant'],
                                 cmap=plt.cm.Blues
                                 )

    plt.show()

    if (features=='original'):
        print('Classification Report for kNN with original features')
        print(classification_report(Y_test, y_pred_test))
    else:
        print('Classification Report for kNN with reduced features')
        print(classification_report(Y_test, y_pred_test))

    return grid

def RandomForestF():
    #LOADING DATASET
    print('Loading Dataset ')
    data = load_breast_cancer()
    dt = pd.DataFrame.from_dict(data["data"])
    dt.columns = data["feature_names"]
    dt["target"] = data["target"]
    X = dt.drop('target', axis=1)
    y = dt['target']

    print("Using Recursive Feature Elimination")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Recursive Feature Elimination
    feature_estimate = RandomForestClassifier(random_state=0)
    feature_estimate.fit(X_train, y_train)
    feature_select = RFECV(feature_estimate , cv=10, step=1, scoring='accuracy')
    feature_select= feature_select.fit(X_train, y_train)
    rfecv = feature_select.get_support()

    ## Append the appropriate features into a list
    reduced_features = []

    for bool, feature in zip(rfecv, X_train.columns):
        if bool:
            reduced_features.append(feature)

    ## Plotting new features set
    print('Optimal number of features :', feature_select.n_features_)
    print('Best features :', reduced_features)
    n_features = X_train.shape[1]
    plt.figure(figsize=(10, 10))
    plt.barh(range(n_features), feature_estimate.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Name of Feature')
    #plt.show()

    ## Train Knn with important features
    X_new = dt.loc[:, reduced_features]
    y_new = dt['target']

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new, y_new, test_size=0.2, random_state=0)

    #Plot accuracies vs k values
    #K_vs_ACC_plot(1, 101, X_train2, y_train2)
    
    model= svm.SVC()
    print(model.get_params().keys())
    param_grid = {'C': np.arange(1, 30), 'kernel': ['linear','poly','rbf','sigmoid']}

    
    #model = KNeighborsClassifier()
    #print(model.get_params().keys())
    #param_grid = {'n_neighbors': np.arange(1, 30), 'weights': ["uniform", "distance"]}
    
    
    #model = RandomForestClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30), 'n_jobs': np.arange(1, 30)}
    
    
    #model = DecisionTreeClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30)}

    #model = LogisticRegression()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30)}
    
    
    clff = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    clff.fit(X_train2, y_train2)
    print('Best k: ', clff.best_params_, ', Best score: ', clff.best_score_ * 100, "%")



    # Running with best parameters
    # Running with best parameters
    clff_best = svm.SVC(**clff.best_params_)
    #clff_best = KNeighborsClassifier(**clff.best_params_)
    #clff_best = RandomForestClassifier(**clff.best_params_)
    #clff_best = DecisionTreeClassifier(**clff.best_params_)
    #clff_best = LogisticRegression(**clff.best_params_)
    
    
    start_time = time.time()
    # Train the model using the training sets
    clff_best.fit(X_train2, y_train2)
    end_time = time.time()
    print("Time spent in Training=", end_time - start_time)
    scores = cross_val_score(clff_best , X_train2, y_train2, cv=10)
    print("Average Cross validation accuracy=", np.sum(scores) / 10)

    # Predictions
    y_pred = clff_best.predict(X_test2)

    cm_test = confusion_matrix(y_test2, y_pred)
    accuracy = accuracy_score(y_test2, y_pred)* 100
    print("Test Score: {} ".format(accuracy))

    '''print("Confusion Matrix Test for kNN with improved features: ", cm_test)
    disp = plot_confusion_matrix(knn4, X_test2,y_test2,
                                 display_labels=['Benign' ,'Malignant'],
                                 cmap=plt.cm.Blues
                                 )
    plt.show()
    '''

    print('Classification Report for kNN with improved features')
    print(classification_report(y_test2, y_pred))
    return accuracy

def L1RF_NB():
    print('Loading Dataset ')
    data = load_breast_cancer()
    dt = pd.DataFrame.from_dict(data["data"])
    dt.columns = data["feature_names"]
    dt["target"] = data["target"]
    X = dt.drop('target', axis=1)
    y = dt['target']


    print("Using L1 Regularization for feature extraction")
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X,y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    # classification on reduced features
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_new,y, test_size=0.2)
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.transform(X_test2)

    gauss = GaussianNB()
    start_reduced = time.time()
    model_class_red = gauss.fit(X_train2,Y_train2)
    stop_reduced = time.time()
    #cross_val_score_red = cross_val_score(model_class_red,x_train_red,y_train_red,cv=10)
    
    red_test = gauss.predict(X_test2)
    acc_red = accuracy_score(Y_test2,red_test)
    #print('Accuracy: ',np.mean(cross_val_score_red))
    print('Reduced Feature set')
    print('Traning time: ',(stop_reduced-start_reduced))
    print('Accuracy: ',acc_red)
    
def L1RF():
    #LOADING DATASET
    print('Loading Dataset ')
    data = load_breast_cancer()
    dt = pd.DataFrame.from_dict(data["data"])
    dt.columns = data["feature_names"]
    dt["target"] = data["target"]
    X = dt.drop('target', axis=1)
    y = dt['target']

    print("Using L1 Regularization for feature extraction")
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X,y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    # classification on reduced features
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_new,y, test_size=0.2)
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.transform(X_test2)
  
    
    
    #model= svm.SVC()
    #print(model.get_params().keys())
    #param_grid = {'C': np.arange(1, 300), 'kernel': ['linear','poly','rbf','sigmoid']}

    
    #model = KNeighborsClassifier()
    #print(model.get_params().keys())
    #param_grid = {'n_neighbors': np.arange(1, 30), 'weights': ["uniform", "distance"]}
    
    
    #model = RandomForestClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30), 'n_jobs': np.arange(1, 30)}
    
    
    #model = DecisionTreeClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30)}

    model = LogisticRegression()
    print(model.get_params().keys())
    param_grid = {'random_state': np.arange(1, 30)}
   
    clff = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    clff.fit(X_test2, Y_test2)
    print('Best k: ', clff.best_params_, ', Best score: ', clff.best_score_ * 100, "%")



    # Running with best parameters
    #clff_best = svm.SVC(**clff.best_params_)
    #clff_best = KNeighborsClassifier(**clff.best_params_)
    #clff_best = RandomForestClassifier(**clff.best_params_)
    #clff_best = DecisionTreeClassifier(**clff.best_params_)
    clff_best = LogisticRegression(**clff.best_params_)
    
    
    start_time = time.time()
    # Train the model using the training sets
    clff_best.fit(X_train2, Y_train2)
    end_time = time.time()
    print("Time spent in Training=", end_time - start_time)
    scores = cross_val_score(clff_best , X_train2, Y_train2, cv=10)
    print("Average Cross validation accuracy=", np.sum(scores) / 10)

    # Predictions
    y_pred = clff_best.predict(X_test2)

    cm_test = confusion_matrix(Y_test2, y_pred)
    accuracy = accuracy_score(Y_test2, y_pred)* 100
    print("Test Score: {} ".format(accuracy))

    print('Classification Report with improved features')
    print(classification_report(Y_test2, y_pred))
    return accuracy

def DTBFS_NB():

    #LOADING DATASET
    print('Loading Dataset ')
    data = load_breast_cancer()
    dt = pd.DataFrame.from_dict(data["data"])
    dt.columns = data["feature_names"]
    dt["target"] = data["target"]
    X = dt.drop('target', axis=1)
    y = dt['target']

    print("Using Tree - based feature selection")

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    #classification on reduced features
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X_new, y, test_size=0.2)
    scaler = StandardScaler()
    X_train3= scaler.fit_transform(X_train3)
    X_test3 = scaler.transform(X_test3)
    
    gauss = GaussianNB()
    start_reduced = time.time()
    model_class_red = gauss.fit(X_train3,Y_train3)
    stop_reduced = time.time()
    #cross_val_score_red = cross_val_score(model_class_red,x_train_red,y_train_red,cv=10)
    
    red_test = gauss.predict(X_test3)
    acc_red = accuracy_score(Y_test3,red_test)
    #print('Accuracy: ',np.mean(cross_val_score_red))
    print('Reduced Feature set')
    print('Traning time: ',(stop_reduced-start_reduced))
    print('Accuracy: ',acc_red)

def DTBFS():
    #LOADING DATASET
    print('Loading Dataset ')
    data = load_breast_cancer()
    dt = pd.DataFrame.from_dict(data["data"])
    dt.columns = data["feature_names"]
    dt["target"] = data["target"]
    X = dt.drop('target', axis=1)
    y = dt['target']

    print("Using Tree - based feature selection")

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    #classification on reduced features
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X_new, y, test_size=0.2)
    scaler = StandardScaler()
    X_train3= scaler.fit_transform(X_train3)
    X_test3 = scaler.transform(X_test3)
    
    
    #model= svm.SVC()
    #print(model.get_params().keys())
    #param_grid = {'C': np.arange(1, 300), 'kernel': ['linear','poly','rbf','sigmoid']}

    
    #model = KNeighborsClassifier()
    #print(model.get_params().keys())
    #param_grid = {'n_neighbors': np.arange(1, 30), 'weights': ["uniform", "distance"]}
    
    
    #model = RandomForestClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30), 'n_jobs': np.arange(1, 30)}
    
    
    #model = DecisionTreeClassifier()
    #print(model.get_params().keys())
    #param_grid = {'random_state': np.arange(1, 30)}

    model = LogisticRegression()
    print(model.get_params().keys())
    param_grid = {'random_state': np.arange(1, 30)}
   
    clff = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    clff.fit(X_test3, Y_test3)
    print('Best k: ', clff.best_params_, ', Best score: ', clff.best_score_ * 100, "%")



    # Running with best parameters
    #clff_best = svm.SVC(**clff.best_params_)
    #clff_best = KNeighborsClassifier(**clff.best_params_)
    #clff_best = RandomForestClassifier(**clff.best_params_)
    #clff_best = DecisionTreeClassifier(**clff.best_params_)
    clff_best = LogisticRegression(**clff.best_params_)
    
    
    start_time = time.time()
    # Train the model using the training sets
    clff_best.fit(X_train3, Y_train3)
    end_time = time.time()
    print("Time spent in Training=", end_time - start_time)
    scores = cross_val_score(clff_best , X_train3, Y_train3, cv=10)
    print("Average Cross validation accuracy=", np.sum(scores) / 10)

    # Predictions
    y_pred = clff_best.predict(X_test3)

    cm_test = confusion_matrix(Y_test3, y_pred)
    accuracy = accuracy_score(Y_test3, y_pred)* 100
    print("Test Score: {} ".format(accuracy))

    print('Classification Report with improved features')
    print(classification_report(Y_test3, y_pred))
    return accuracy

    
if __name__ == '__main__':
    ac = DTBFS()
    print(ac)