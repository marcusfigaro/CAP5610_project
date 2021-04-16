import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn. model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import time

data = pd.read_csv('data.csv')

# Original Feature Set
x = data.drop(columns=['id','diagnosis','Unnamed: 32'])
y = np.ravel(data[['diagnosis']])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#Recursive Feature Elimination
feature_estimate = RandomForestClassifier(random_state=0)
feature_estimate.fit(x_train, y_train)
feature_select = RFECV(feature_estimate , cv=10, step=1, scoring='accuracy')
feature_select= feature_select.fit(x_train, y_train)
rfecv = feature_select.get_support()

## Append the appropriate features into a list
reduced_features = []

for bool, feature in zip(rfecv, x_train.columns):
    if bool:
        reduced_features.append(feature)

# Testing on original dataset
gauss = GaussianNB()
start = time.time()
model_class = gauss.fit(x_train,y_train)
stop = time.time()
#cross_val_score = cross_val_score(model_class,x_train,y_train,cv=10)
#print('Accuracy: ',np.mean(cross_val_score))
print('Default dataset')
print("Training time: ",(stop-start))
y_test_norm = gauss.predict(x_test)
acc_norm = accuracy_score(y_test,y_test_norm)
print('Accuracy: ',acc_norm)

# Testing on reduced feature set
x_red = data.loc[:,reduced_features]
y_red = np.ravel(data[['diagnosis']])

x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(x_red,y_red,test_size=0.2)

gauss = GaussianNB()
start_reduced = time.time()
model_class_red = gauss.fit(x_train_red,y_train_red)
stop_reduced = time.time()
#cross_val_score_red = cross_val_score(model_class_red,x_train_red,y_train_red,cv=10)

red_test = gauss.predict(x_test_red)
acc_red = accuracy_score(y_test_red,red_test)
#print('Accuracy: ',np.mean(cross_val_score_red))
print('Reduced Feature set')
print('Traning time: ',(stop_reduced-start_reduced))
print('Accuracy: ',acc_red)