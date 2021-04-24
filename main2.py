'''
Created on Mar 23, 2021

@author: 14076
'''
from Kmeans import kmeans
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from KNN import *
from sklearn.preprocessing import StandardScaler
from Kmeans import compute_distance
from Kmeans import calc_avg_dist
from Kmeans import min_centroid_dist
from Kmeans import calc_fuzzy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


if __name__ == '__main__':
    print('Loading Dataset ')
    data = np.loadtxt("data.csv", float, delimiter=',', skiprows=1, usecols=range(2, 32))
    labels = np.loadtxt("data.csv", str, delimiter=',', skiprows=1, usecols=(1))
    labels = labels.reshape((len(labels), 1))

    df=pd.DataFrame(labels, columns=['label'])
    #sns.countplot(df["label"])
    plt.show()

    print('Separating Benign and Malignant Tumors based on labels')

    malignant = np.empty((0, data.shape[1]))
    benign = np.empty((0, data.shape[1]))

    for i in range(len(data)):
        if labels[i] == 'M':
            malignant = np.append(malignant, [data[i]], axis=0)
        else:
            benign = np.append(benign, [data[i]], axis=0)

    print('Saving Malignant and Benign data individually')

    features = "radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst,"

    np.savetxt("data/malignant.csv",malignant,delimiter=',',header=features)
    np.savetxt("data/benign.csv",benign,delimiter=',',header=features)

    malignant = np.loadtxt("data/malignant.csv", delimiter=',')
    benign=np.loadtxt("data/benign.csv", delimiter=',')

    no_of_clusters=3  # still need to find optimum number of cluster by finding validity ratio

    print('Performing K-means on Malignant Tumors')
    M_clusters,M_centroids=kmeans(malignant,no_of_clusters)

    print('Performing K-means on Benign Tumors')
    B_clusters,B_centroids=kmeans(benign,no_of_clusters)


    k = 1
    print("Saving cluster centroids and points")
    k = 1
    for cluster_no, values in M_clusters.items():
        points = []
        for i in values:
            points.append(i.tolist())
        filename = 'Malignant_points' + str(k) + '.csv'
        np.savetxt("data/"+filename, np.array(values))
        k = k + 1
    np.savetxt('data/Malignant_centroids.csv', np.array(M_centroids))

    k = 1
    for cluster_no, values in B_clusters.items():
        points = []
        for i in values:
            points.append(i.tolist())
        filename = 'Benign_points' + str(k) + '.csv'
        np.savetxt("data/"+filename, np.array(values))
        k = k + 1
    np.savetxt('data/Benign_centroids.csv', np.array(B_centroids))

    #To laod files again:
    B1 = np.loadtxt("data/Benign_points1.csv", delimiter=' ', dtype=float)
    B2 = np.loadtxt("data/Benign_points2.csv", delimiter=' ', dtype=float)
    B3 = np.loadtxt("data/Benign_points3.csv", delimiter=' ', dtype=float)
    M1 = np.loadtxt("data/Malignant_points1.csv", delimiter=' ', dtype=float)
    M2 = np.loadtxt("data/Malignant_points2.csv", delimiter=' ', dtype=float)
    M3 = np.loadtxt("data/Malignant_points3.csv", delimiter=' ', dtype=float)

    B_centroids_ben = np.loadtxt("data/Benign_centroids.csv", delimiter=' ', dtype=float)
    M_centroids_mal = np.loadtxt("data/Malignant_centroids.csv", delimiter=' ', dtype=float)

    points_list = []
    points_list.append(M1)
    points_list.append(M2)
    points_list.append(M3)
    points_list.append(B1)
    points_list.append(B2)
    points_list.append(B3)

    ## Calculate validity ratio
    # Malignant
    min_dist_mal = min_centroid_dist(M_centroids_mal)
    j = 0
    dist_m = []
    for i in range(1, no_of_clusters + 1):
        dist_m.append(calc_avg_dist(M_clusters[i], M_centroids_mal[j]))
        j = j + 1
    avg_dist_m = np.mean(dist_m)
    val_ratio_m = avg_dist_m / min_dist_mal
    print("Calculating validity ratio for Malignant k-means")
    print(val_ratio_m)

    # Benign
    min_dist_ben = min_centroid_dist(B_centroids_ben)
    j = 0
    dist_b = []
    for i in range(1, no_of_clusters + 1):
        dist_b.append(calc_avg_dist(B_clusters[i], B_centroids_ben[j]))
        j = j + 1
    avg_dist_b = np.mean(dist_b)
    val_ratio_b = avg_dist_b / min_dist_ben
    print("Calculating validity ratio for Benign k-means")
    print(val_ratio_b)

    ## Membership Functions
    all_points = np.concatenate((malignant, benign))
    all_centroids = np.concatenate((M_centroids_mal, B_centroids_ben))
    fuzz_mem = []
    fuzz_num, fuzz_den = calc_fuzzy(all_points, all_centroids, points_list)

    fuzz_df = {}
    for i in range(len(points_list)):
        fuzz_df[i] = 1 - fuzz_num[i] / (30 * fuzz_den[i])
    fuzz_df = pd.DataFrame(fuzz_df)

    fuzz_df.to_csv('membership_values.csv')


    ##SVM with reduced features =6
    def read_dataset(fname):
        data = pd.read_csv(fname, index_col=0)
        data = data.fillna(0)
        return data

    def muti_score(model):
        warnings.filterwarnings('ignore')

        precision = cross_val_score(model, X, y, scoring='accuracy', cv=10)

        print("accuracy",precision.mean())

    train = read_dataset('data/membership_values.csv')

    train.insert(0, 'type', 'M')

    counter=0

    for i in range(1,569):
        counter +=1
        if(counter>=212):
            train.loc[counter, 'type'] = 'B'


    train=shuffle(train)
    train=train.reset_index(drop=True)
    y = train['type'].values
    X = train.drop(['type'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    print("Using PCA for feature extraction")
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(data, labels, test_size=0.2)
    scaler = StandardScaler()
    X_train1 = scaler.fit_transform(X_train1)
    X_test1 = scaler.transform(X_test1)
    pca = PCA(n_components=10, svd_solver='full')
    X_train1 = pca.fit_transform(X_train1)
    X_test1 = pca.transform(X_test1)
    pca.explained_variance_ratio_
    classifier = svm.SVC(C = 10, kernel = "linear",gamma=0.001)
    classifier.fit(X_train1, Y_train1)
    y_pred = classifier.predict(X_test1)
    print(metrics.classification_report(Y_test1, y_pred))

    temp=0
    for i in range(0,569):
        if (labels[temp]=='B'):
            labels[temp]=0
        if (labels[temp]=='M'):
            labels[temp] = 1
        temp += 1


    print("Using Lasso  for feature extraction")
    from sklearn.linear_model import LassoCV
    scaler = preprocessing.MinMaxScaler()
    X_train_minmax = scaler.fit_transform(X_train1)
    X_test_minmax = scaler.fit_transform(X_test1)
    X_minmax = scaler.fit_transform(data)
    clf = LassoCV().fit(X_minmax, labels)
    importance = np.abs(clf.coef_)
    print(importance)




