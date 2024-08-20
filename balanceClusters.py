import numpy as np

def balanceClusters(clusters, train):
    noOfClasses = np.unique(train[:, -1])
    b_index = 0
    balancedClusters = []
    centroids = []

    for i in range(len(clusters)):
        balancedCluster = clusters[i]['train']
        majorClass = np.unique(clusters[i]['train'][:, -1])
        centroids.append(clusters[i]['centroid'])

        for j in range(len(noOfClasses)):
            if majorClass != noOfClasses[j]:
                records = len(clusters[i]['train'])
                toAdd = train[train[:, -1] == noOfClasses[j], :]

                closestToCentroid = []
                for k in range(len(toAdd)):
                    closestToCentroid.append(np.linalg.norm(clusters[i]['centroid'] - toAdd[k, :-1]))

                sorted_ids = np.argsort(closestToCentroid)
                toAdd = toAdd[sorted_ids, :]

                if len(toAdd) >= records:
                    balancedCluster = np.vstack((balancedCluster, toAdd[:records, :]))
                else:
                    balancedCluster = np.vstack((balancedCluster, toAdd))

        balancedClusters.append(balancedCluster)
        b_index += 1

    return balancedClusters, centroids
