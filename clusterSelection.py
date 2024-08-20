from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import mode
from pyswarm import pso
from sklearn.metrics import accuracy_score
from trainClassifiers import trainClassifiers
import copy

def generateClusters(train):
    totalClusters = 0
    genClusters = []
    noOfIterations = round(np.power(len(train), 1/5))
    for clusters in range(1, noOfIterations + 1):
        kmeans = KMeans(n_clusters=clusters, max_iter=24000).fit(train)
        for j in range(clusters):
            clusterData = train[kmeans.labels_ == j, :]
            unique_y_values = np.unique(clusterData[:, -1])    #for SVM, else gives homogenous class error
            if len(unique_y_values) == 1:
                continue
            genClusters.append(clusterData)
            totalClusters += 1
    print("totalClusters : ", totalClusters)
    return genClusters

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
            classifiers = trainClassifiers(allClusters[j][:, :-1], allClusters[j][:, -1], clusteringParams)
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

def clusterSelection(trainX, trainy, valX, valy, params):
    allClusters = generateClusters(np.column_stack((trainX, trainy)))
    bestClusters = clusteringPSO(allClusters, np.column_stack((valX, valy)), params)
    bestClusters = np.flatnonzero(bestClusters['chromosome'])
    print("bestClusters: ",bestClusters)
    selectedClusters = [allClusters[i] for i in bestClusters]
    return selectedClusters