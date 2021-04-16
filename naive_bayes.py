import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn. model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
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


gauss = GaussianNB()
start = time.time()
model_class = gauss.fit(x_train,y_train)
stop = time.time()
score = cross_val_score(model_class,x,y,cv=10)
print('Accuracy: ',np.mean(score))
print("Training time: ",(stop-start))

x_red = data.loc[:,reduced_features]
y_red = np.ravel(data[['diagnosis']])

gauss = GaussianNB()
start_reduced = time.time()
model_class_red = gauss.fit(x_red,y_red)
stop_reduced = time.time()
score_red = cross_val_score(model_class_red,x_red,y_red,cv=10)
print('Accuracy: ',np.mean(score_red))
print('Traning time: ',(stop_reduced-start_reduced))