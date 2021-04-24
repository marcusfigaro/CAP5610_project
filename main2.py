
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

if __name__ == '__main__':

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
    classifier = svm.SVC(C = 10, kernel = "linear",gamma=0.001)
    classifier.fit(X_train2, Y_train2)
    y_pred = classifier.predict(X_test2)
    print(metrics.classification_report(Y_test2, y_pred))


    print("Using Tree - based feature selection")
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
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
    classifier = svm.SVC(C=10, kernel="linear", gamma=0.001)
    classifier.fit(X_train3, Y_train3)
    y_pred = classifier.predict(X_test3)
    print(metrics.classification_report(Y_test3, y_pred))






