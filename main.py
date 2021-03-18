from Kmeans import kmeans
from Kmeans import compute_distance
from Kmeans import calc_avg_dist
from Kmeans import min_centroid_dist
from Kmeans import calc_fuzzy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


if __name__ == '__main__':
    print('Loading Dataset ')
    data = np.loadtxt("data.csv", float, delimiter=',', skiprows=1, usecols=range(2, 32))
    labels = np.loadtxt("data.csv", str, delimiter=',', skiprows=1, usecols=(1))
    labels = labels.reshape((len(labels), 1))

    df=pd.DataFrame(labels, columns=['label'])
    sns.countplot(df["label"])
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

    malignant = np.loadtxt("malignant.csv", delimiter=',')
    benign=np.loadtxt("benign.csv", delimiter=',')

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
    B1 = np.loadtxt("Benign_points1.csv", delimiter=' ', dtype=float)
    B2 = np.loadtxt("Benign_points2.csv", delimiter=' ', dtype=float)
    B3 = np.loadtxt("Benign_points3.csv", delimiter=' ', dtype=float)
    M1 = np.loadtxt("Malignant_points1.csv", delimiter=' ', dtype=float)
    M2 = np.loadtxt("Malignant_points2.csv", delimiter=' ', dtype=float)
    M3 = np.loadtxt("Malignant_points3.csv", delimiter=' ', dtype=float)

    B_centroids_ben = np.loadtxt("Benign_centroids.csv", delimiter=' ', dtype=float)
    M_centroids_mal = np.loadtxt("Malignant_centroids.csv", delimiter=' ', dtype=float)

    ## Calculate validity ratio
    # Malignant
    min_dist = min_centroid_dist(M_centroids_mal)
    j=0
    dist_m = []
    for i in range(1,no_of_clusters+1):
        dist_m.append(calc_avg_dist(M_clusters[i],M_centroids_mal[j]))
        j = j+1
    avg_dist = np.mean(dist_m)
    val_ratio_m = avg_dist/min_dist
    print("Calculating validity ratio for Malignant k-means")
    print(val_ratio_m)

    # Benign
    min_dist = min_centroid_dist(B_centroids_ben)
    j = 0
    dist_b = []
    for i in range(1, no_of_clusters + 1):
        dist_b.append(calc_avg_dist(B_clusters[i], B_centroids_ben[j]))
        j = j + 1
    avg_dist = np.mean(dist_b)
    val_ratio_b = avg_dist / min_dist
    print("Calculating validity ratio for Benign k-means")
    print(val_ratio_b)

    ## Membership Functions
    all_points = np.concatenate((malignant,benign))
    all_centroids = np.concatenate((M_centroids_mal,B_centroids_ben))
    fuzz_mem = []
    for j in all_centroids:
        fuzz_mem.append(calc_fuzzy(all_points,j))
    fuzz_df = pd.DataFrame(fuzz_mem)
    fuzz_df = fuzz_df.T
    fuzz_df.to_csv("data/membership_values.csv")


'''
For classification, columns of fuzz_df dataframe (membership_values csv file) are new features

TO DO: 
3. Do classification (SVM)
 
'''




