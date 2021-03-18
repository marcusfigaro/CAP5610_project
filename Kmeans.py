import random
import numpy as np

def kmeans(data, no_of_clusters):

    centroids=find_centroid(data, no_of_clusters)

    max_iterations=100
    iteration=0
    old_centroids = centroids.copy()

    while 1:
        new_centroids = []
        cluster_points = reassignment_of_points(data, old_centroids,no_of_clusters)

        new_centroids = reassign_centroids(cluster_points)
        for cluster_no, values in cluster_points.items():
            points = []
            for i in values:
                points.append(i.tolist())

        iteration += 1

        if check_convergence(old_centroids , new_centroids , iteration, max_iterations) is True:
            break;

        old_centroids = new_centroids.copy()

    print('K-means converged after {} iterations'.format(iteration))
    return cluster_points, new_centroids


def find_centroid(data,no_of_clusters):
    print('Generating random centroids')
    index = []
    centroid_list = []
    for i in range(1,no_of_clusters+1):
        index.append(random.randint(0, data.shape[0] - 1))

    for j in index:
        centroid_list.append(data[j])

    return centroid_list

def reassignment_of_points(data, old_centroids, no_of_clusters):
    cluster_points = {}

    for i in range(1, no_of_clusters + 1):
        cluster_points.setdefault(i, [])

    for data_point in data:
        distance_list = []
        for centroid in old_centroids:
            distance_list.append(compute_distance(data_point, centroid,'euclidean'))
        distance_list = np.array(distance_list)
        assigned_cluster = np.argmin(distance_list) + 1
        cluster_points[assigned_cluster].append(data_point)

    return cluster_points


def compute_distance(data_point, centroid,dist_type):
        if dist_type == 'euclidean':
            return np.sqrt(np.sum((data_point - centroid) ** 2))

        elif dist_type == 'manhattan':
            return np.sum(np.abs(data_point - centroid))

        elif dist_type == 'cosine':
            return 1 - (np.dot(data_point, centroid) / (np.linalg.norm(data_point) * np.linalg.norm(centroid)))


def reassign_centroids(cluster_points):
    new_centroids = []
    for cluster_no, points in cluster_points.items():
        m = np.mean(np.array(points), axis=0)
        new_centroids.append(m)
    return new_centroids


def check_convergence(old_centroids, new_centroids, iteration, max_iterations):
    error = compute_distance(np.array(old_centroids), np.array(new_centroids),'euclidean')
    #print("error=",error)

    if iteration > max_iterations or error <= 0.00001:
        return True
    return False

def calc_avg_dist(data,centroid):
    dist = []
    for i in data:
        dist.append(compute_distance(i,centroid,'euclidean'))
    dist = np.array(dist)
    avg_dist = np.mean(dist)
    return avg_dist

def min_centroid_dist(centroid_array):
    cen_dist = []
    for i in centroid_array:
        for j in centroid_array:
            cen_dist.append(compute_distance(i,j,'euclidean'))
    cen_dist = np.array(cen_dist)
    cen_dist = cen_dist[cen_dist!=0]
    min_dist = min(cen_dist)
    return min_dist

def calc_validity_ratio(clusters,centroids):
    min_dist = min_centroid_dist(centroids)
    j=0
    dist = []
    for i in range(1,no_of_clusters+1):
        dist_m.append(calc_avg_dist(clusters[i],centroids[j]))
        j = j+1
    avg_dist = np.mean(dist)
    val_ratio = avg_dist/min_dist
    return val_ratio

def calc_fuzzy(data,centroid):
    fuzzy_dist = []
    for i in data:
        fuzzy_dist.append(compute_distance(i,centroid,'manhattan'))
    num5 = np.array(fuzzy_dist)
    den5 = max(num5)
    fuzz_mem = 1-num5/den5
    return fuzz_mem