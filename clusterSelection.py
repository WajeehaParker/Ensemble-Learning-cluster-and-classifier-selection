from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.stats import mode
from pyswarm import pso
#from sklearn.metrics import accuracy_score
from trainClassifiers import trainClassifiers
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import copy
import skfuzzy as fuzz
import time
from helpers import compute_weights, weighted_voting
from pyswarms.discrete.binary import BinaryPSO

########## cluster generation methods start ##########

def generateClusters(train):
    genClusters = []
    noOfIterations = round(np.power(len(train), 1/3))
    totalClustersCount = 0
    for clusters in range(1, noOfIterations + 1):
        if clusters % 3 == 0:
            continue
        kmeans = KMeans(n_clusters=clusters, max_iter=24000).fit(train)
        for j in range(clusters):
            totalClustersCount += 1
            clusterData = train[kmeans.labels_ == j, :]
            unique_y_values = np.unique(clusterData[:, -1])    #remove homogenous clusters (for SVM, else gives homogenous class error)
            if len(unique_y_values) == 1:
                continue
            genClusters.append(clusterData)
    return genClusters, totalClustersCount

def generateFuzzyClusters(train):
    genClusters = []
    noOfIterations = round(np.power(len(train), 1/5))  
    #print("noOfIterations: ", noOfIterations)  
    for clusters in range(1, noOfIterations + 1):
        #print("Executing iteration # ", clusters)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(train[:, :-1].T, clusters, 2, error=0.005, maxiter=1000, init=None)
        for i in range(clusters):
            clusterData = train[u[i] > 0.6]
            unique_y_values = np.unique(clusterData[:, -1])
            if len(unique_y_values) > 1:
                genClusters.append(clusterData)
    #print("Total clusters generated:", len(genClusters))
    return genClusters, clusters

def generateHieraricalClusters(train):
    genClusters = []
    noOfIterations = round(np.power(len(train), 1 / 5))
    totalClustersCount = 0
    
    for clusters in range(1, noOfIterations + 1):
        agglomerative = AgglomerativeClustering(n_clusters=clusters)
        cluster_labels = agglomerative.fit_predict(train)
        for j in range(clusters):
            totalClustersCount += 1
            clusterData = train[cluster_labels == j, :]
            if len(clusterData) <= 10:
                continue
            unique_y_values = np.unique(clusterData[:, -1])
            if len(unique_y_values) == 1:
                continue
            genClusters.append(clusterData)
    print("Total Clusters generated:", len(genClusters))
    return genClusters, totalClustersCount

def generateHierarchicalClustersv2(train, distance_threshold=0):
    genClusters = []
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    cluster_labels = agglomerative.fit_predict(train)
    
    #Some labels may repeat, indicating multiple points belong to the same cluster. Extract the distinct cluster labels, so you can efficiently process each cluster once. Without extracting unique labels, iterating over the raw labels would repeatedly process the same cluster for every data point, resulting in redundant operations.
    #unique_clusters = np.unique(cluster_labels)

    for cluster_id in cluster_labels:
        clusterData = train[cluster_labels == cluster_id, :]
        if len(clusterData) <= 10:
            continue
        unique_y_values = np.unique(clusterData[:, -1])
        if len(unique_y_values) == 1:
            continue
        genClusters.append(clusterData)
    
    print("Total Clusters generated (size > 10):", len(genClusters))
    return genClusters

def generateEnsembleClusters(train):
    clustering_algorithms = [
        KMeans(n_clusters=5, random_state=0),
        AgglomerativeClustering(n_clusters=None, distance_threshold=25),
        DBSCAN(eps=1.5, min_samples=5),
    ]
    
    n_samples = train.shape[0]
    co_association_matrix = np.zeros((n_samples, n_samples))
    min_cluster_size=10

    for algorithm in clustering_algorithms:
        labels = algorithm.fit_predict(train)
        for i in range(n_samples):
            for j in range(n_samples):
                co_association_matrix[i, j] += 1

    # Normalize the co-association matrix
    co_association_matrix /= len(clustering_algorithms)

    # Ensure diagonal is 1 (self-similarity)
    np.fill_diagonal(co_association_matrix, 1)

    # Convert the co-association matrix to distance format
    distance_matrix = 1 - co_association_matrix
    condensed_distance_matrix = squareform(distance_matrix)

    # Use hierarchical clustering on the co-association matrix
    linkage_matrix = linkage(condensed_distance_matrix, method='average')
    cluster_labels = fcluster(linkage_matrix, t=0.6, criterion='distance')  # Adjust threshold as needed

    # Extract clusters
    genClusters = []
    unique_labels = np.unique(cluster_labels)
    for cluster_id in unique_labels:
        clusterData = train[cluster_labels == cluster_id, :]
        if len(clusterData) >= min_cluster_size:
            unique_y_values = np.unique(clusterData[:, -1])
            if len(unique_y_values) > 1:  # Exclude homogeneous clusters
                genClusters.append(clusterData)
    
    print("Total Clusters generated (size > {}):".format(min_cluster_size), len(genClusters))
    return genClusters

def generateClustersUsingElbow(train):
    max_clusters = 10
    wcss = []
    
    # Calculate WCSS scores for different numbers of clusters.  WCSS (Within-Cluster Sum of Square) is the sum of the squared distance between each point and the centroid in a cluster
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(train)
        wcss.append(kmeans.inertia_) 

    optimal_clusters = np.argmin(np.gradient(np.gradient(wcss))) + 2
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(train)

    genClusters = []
    for cluster_id in range(optimal_clusters):
        clusterData = train[cluster_labels == cluster_id, :]
        if len(clusterData) >= 10:
            genClusters.append(clusterData)
    return genClusters, optimal_clusters

########## cluster generation methods end ##########



def clusteringPSO(allClusters, testData, params):
    def PSOAF(c):
        c = c > 0.6                                             # threshold for selecting classifier on the basis of PSO particle position
        c = np.where(c)[0]                                      # retrieve selected particles
        decisionMatrix = np.ones((len(testData[:, -1]), len(c)))
        for i in range(len(c)):
            decisionMatrix[:, i] = allPredictions[:, c[i]]
        decisionMatrix = mode(decisionMatrix, axis=1)[0]        # majority voting
        error = np.mean(decisionMatrix != testData[:, -1])
        return error

    try:
        allPredictions = np.zeros((len(testData), len(allClusters)))
        clusteringParams = copy.deepcopy(params)
        clusteringParams['classifiers'] = clusteringParams['classifiers'][:1]    #first classifier in the list will be used for cluster selection
        for j in range(len(allClusters)):
            #start_time = time.time()
            classifiers = trainClassifiers(allClusters[j][:, :-1], allClusters[j][:, -1], clusteringParams)
            #end_time = time.time()
            #duration = end_time - start_time
            #minutes = int(duration // 60) 
            #seconds = int(duration % 60) 
            #TimeForPSO1 = f"{minutes}m {seconds}s"
            #print("Cluster ",j,": ", TimeForPSO1)
            prediction = classifiers[0]['model'].predict(testData[:, :-1])
            allPredictions[:, j] = prediction
            #accuracy = accuracy_score(testData[:, -1], predictions)

        lb = np.zeros(allPredictions.shape[1])
        ub = np.ones(allPredictions.shape[1])
        best, fval = pso(PSOAF, lb, ub, swarmsize=50)   # max_iter=100 (typical default for many PSO implementations)   # Number of evaluations = 50×100 = 5000. PSOAF will run 5000 times
        obj = {
            'chromosome': np.round(best),
            'fval': fval
        }
    except Exception as exc:
        print(f'Problem with {exc}')
        obj = None

    return obj

def clusteringPSO2(allClusters, testData, params):
    try:
        # Generate predictions from each cluster's first classifier
        allPredictions = np.zeros((len(testData), len(allClusters)))
        clusteringParams = copy.deepcopy(params)
        clusteringParams['classifiers'] = clusteringParams['classifiers'][:1]  # Use first classifier for cluster selection
        
        for j in range(len(allClusters)):
            classifiers = trainClassifiers(allClusters[j][:, :-1], allClusters[j][:, -1], clusteringParams)
            prediction = classifiers[0]['model'].predict(testData[:, :-1])
            allPredictions[:, j] = prediction
        
        # Prepare true labels
        y_true = testData[:, -1]
        n_classifiers = allPredictions.shape[1]

        # Define the objective function for PSO
        def objective_function(swarm):
            n_particles = swarm.shape[0]
            costs = np.zeros(n_particles)
            for i in range(n_particles):
                selected = swarm[i, :].astype(bool)
                if np.sum(selected) == 0:
                    costs[i] = 1.0  # Handle case with no classifiers selected
                    continue
                predictions_subset = allPredictions[:, selected]
                # Compute majority vote
                # Efficient majority voting using vectorization
                # unique_classes = np.unique(predictions_subset)
                # if len(unique_classes) == 0:
                #     majority_vote = np.zeros(predictions_subset.shape[0], dtype=int)
                # else:
                #     # Reshape for broadcasting and compute counts
                #     matches = (predictions_subset[:, :, np.newaxis] == unique_classes)
                #     counts = matches.sum(axis=1)
                #     max_indices = counts.argmax(axis=1)
                #     majority_vote = unique_classes[max_indices]
                
                # Compute weighted voting
                weights = compute_weights(predictions_subset, y_true)
                majority_vote = weighted_voting(predictions_subset, weights, y_true)
                
                accuracy = np.mean(majority_vote == y_true)
                costs[i] = 1.0 - accuracy  # Minimize 1 - accuracy
            return costs

        # PSO parameters (can be adjusted via params if needed)
        pso_options = params.get('pso_options', {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 5, 'p': 1})
        n_particles = params.get('n_particles', 50)
        iterations = params.get('iterations', 100)

        # Create initial positions where one particle has all classifiers selected
        init_pos = np.zeros((n_particles, n_classifiers), dtype=int)  # Initialize to 0s
        init_pos[0, :] = 1  # First particle selects all classifiers (all positions = 1)
        # Randomly initialize the rest of the particles
        for i in range(1, n_particles):
            init_pos[i] = np.random.randint(2, size=n_classifiers)

        # Initialize and run BinaryPSO
        optimizer = BinaryPSO(n_particles=n_particles, dimensions=n_classifiers, 
                              options=pso_options, init_pos=init_pos)

        # Run optimization with custom initial positions
        cost, pos = optimizer.optimize(
            objective_function,
            iters=iterations,
            # init_pos=init_pos  # Pass custom initial positions
        )

        # Determine the best combination
        selected_indices = np.where(pos)[0].tolist()
        if not selected_indices:  # Fallback if no classifiers selected
            selected_indices = [0]
        # best_predictions = allPredictions[:, selected_indices]
        # majority_vote = scipy.stats.mode(best_predictions, axis=1, keepdims=False).mode
        # best_accuracy = np.mean(majority_vote == y_true)
        # print(f'Best accuracy: {best_accuracy:.4f}')

    except Exception as exc:
        print(f'Problem with {exc}')
        selected_indices = []
        best_accuracy = 0.0

    return selected_indices

def clusterSelection(trainX, trainy, valX, valy, params, X_test, y_test):
    clusteringInfo = {}
    #print("Generating Clusters at: ", datetime.now())
    genClusters, totalClustersCount = generateClusters(np.column_stack((trainX, trainy)))
    #genClusters, totalClustersCount = generateFuzzyClusters(np.column_stack((trainX, trainy)))
    #genClusters, totalClustersCount = generateHieraricalClusters(np.column_stack((trainX, trainy)))
    #genClusters = generateHierarchicalClustersv2(np.column_stack((trainX, trainy)),40)
    #genClusters = generateEnsembleClusters(np.column_stack((trainX, trainy)))
    #genClusters, totalClustersCount = generateClustersUsingElbow(np.column_stack((trainX, trainy)))
    print("totalClustersCount:",totalClustersCount," selectedClusterCount:",len(genClusters))
    #print("Cluster generation completed at: ", datetime.now())
    print("Applying PSO on Clusters at: ",datetime.now())
    start_time = time.time()
    #bestClusters = clusteringPSO(genClusters, np.column_stack((valX, valy)), params)
    bestClusters = clusteringPSO2(genClusters, np.column_stack((valX, valy)), params)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60) 
    seconds = int(duration % 60) 
    print("PSO completed at: ", datetime.now())
    #bestClusters = np.flatnonzero(bestClusters['chromosome'])
    selectedClusters = [genClusters[i] for i in bestClusters]
    #selectedClusters=genClusters
    clusteringInfo['TotalClustersCount'] = totalClustersCount
    clusteringInfo['NonHomogenousClustersCount'] = len(genClusters)
    clusteringInfo['ClustersSelectedByPSO'] = len(selectedClusters)
    clusteringInfo['TimeForPSO1'] = f"{minutes}m {seconds}s"
    #clusteringInfo['ClustersSelectedByPSO'] = 0
    #clusteringInfo['TimeForPSO1'] = 0

    # train Decision Tree classifiers on selected clusters
    classifiers = []
    clusteringParams = copy.deepcopy(params)
    clusteringParams['classifiers'] = clusteringParams['classifiers'][:1]  # Use first classifier for cluster selection
        
    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        all = trainClassifiers(X, y, clusteringParams)
        classifiers.extend(all)

    # apply the trained classifiers on test data
    decisionMatrix = np.zeros((len(X_test), len(classifiers)))
    index = 0
    for i in range(len(classifiers)):
        try:
            decisionMatrix[:, index] = classifiers[i]['model'].predict(X_test)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')
    
    decisionMatrix_val = np.ones((len(valX), len(classifiers)))
    index = 0
    for i in range(len(classifiers)):
        try:
            decisionMatrix_val[:, index] = classifiers[i]['model'].predict(valX)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')

    weights = compute_weights(decisionMatrix_val, valy)

    # fusion with weighted majority voting
    decisionMatrix = weighted_voting(decisionMatrix, weights, y_test)
    acc = np.mean(decisionMatrix == y_test)
    print(f"Accuracy after Stage 1: {acc}")
    #clusteringInfo['AccAfterStage1'] = acc

    return selectedClusters, clusteringInfo