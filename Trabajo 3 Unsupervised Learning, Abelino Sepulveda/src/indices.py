import numpy as np
from sklearn import metrics

def intra_cluster(data, clusters):
    labels = np.zeros(len(data))
    group = 0
    cont = 0
    for _, cluster in clusters.items():
        labels[cluster] = group
        if len(cluster)>0:
            cont+=1
        group += 1

    if cont>1 and len(clusters.keys())<data.shape[0]:
        silhouette_score = metrics.silhouette_score(data, labels)
        davies_bouldin_index = metrics.davies_bouldin_score(data, labels)
        return {'silhouette_score':silhouette_score, 'davies_bouldin_index': davies_bouldin_index }
    else: 
        return 'un solo cluster'

def extra_cluster(data, clusters):

    clusters_labels = np.zeros(len(data))
    group = 0
    cont = 0
    for _, cluster in clusters.items():
        clusters_labels[cluster] = group
        if len(cluster)>0:
            cont+=1
        group += 1

    if cont>1 and len(clusters.keys())<data.shape[0]:
        calinski_harabasz_index = metrics.calinski_harabasz_score(data, clusters_labels)
        return  {'calinski_harabasz_index':calinski_harabasz_index}
    else:
        return 'un solo cluster'