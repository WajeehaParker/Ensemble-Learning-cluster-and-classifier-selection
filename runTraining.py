import os
import numpy as np
from clusterSelection import clusterSelection
from classifierSelection import classifierSelection, classifierSelectionWithSHAP
from helpers import compute_weights, weighted_voting
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

def readData(p_name):
    file_path = os.path.join('DTE', p_name, 'data.csv')
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Adjust delimiter if necessary
    X = data[:, :-1]
    Y = data[:, -1]
    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # Adjust test_size if needed
    trainX, valX, trainy, valy = train_test_split(X, Y, test_size=0.25, random_state=42)  # Adjust test_size if needed
    return trainX, trainy, valX, valy, X_test, y_test

def fusion_majorityVoting(classifiers, data, valX, valy):
    X = data[:, :-1]
    Y = data[:, -1]
    decisionMatrix = np.ones((len(X), len(classifiers)))
    index = 0
    
    for i in range(len(classifiers)):
        try:
            decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')
    
    decisionMatrix = mode(decisionMatrix, axis=1)[0]
    acc = np.mean(decisionMatrix == Y)
    print(f"Accuracy: {acc}")
    return acc

def fusion_weightedVoting(classifiers, data, valX, valy):
    X = data[:, :-1]
    Y = data[:, -1]
    decisionMatrix = np.ones((len(X), len(classifiers)))
    index = 0
    
    for i in range(len(classifiers)):
        try:
            decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')
    
    #decisionMatrix = mode(decisionMatrix, axis=1)[0]
    decisionMatrix = apply_weighted_voting(decisionMatrix, classifiers, valX, valy)
    acc = np.mean(decisionMatrix == Y)
    f1 = f1_score(Y, decisionMatrix, average='weighted')
    precision = precision_score(Y, decisionMatrix, average='weighted')
    recall = recall_score(Y, decisionMatrix, average='weighted')
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")
    return acc

def apply_weighted_voting(decisionMatrix, classifiers, valX, valY):
    decisionMatrix_val = np.ones((len(valX), len(classifiers)))
    index = 0
    for i in range(len(classifiers)):
        try:
            decisionMatrix_val[:, index] = classifiers[i]['model'].predict(valX)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')
    weights = compute_weights(decisionMatrix_val, valY)
    return weighted_voting(decisionMatrix, weights, valY)

def runTraining(p_name, params):
    results = {}
    nonOptimized_Accuracy = []
    optimized_Accuracy = []
    
    print(p_name)
    trainX, trainy, valX, valy, X_test, y_test = readData(p_name)
    
    selectedClusters, clusteringInfo = clusterSelection(trainX, trainy, valX, valy, params, X_test, y_test)
    print("Cluster selection completed")
    classifiers, selectedClassifiers, TimeForPSO2 = classifierSelection(selectedClusters, valX, valy, params)
    print("Classifier selection completed")

    nonOptimized_Accuracy.append(fusion_weightedVoting(classifiers, np.column_stack((X_test, y_test)), valX, valy))
    optimized_Accuracy.append(fusion_weightedVoting(selectedClassifiers, np.column_stack((X_test, y_test)), valX, valy))
    #end for

    results['p_name'] = p_name
    results['TotalClustersCount'] = clusteringInfo['TotalClustersCount']
    results['NonHomogenousClustersCount'] = clusteringInfo['NonHomogenousClustersCount']
    results['ClustersSelectedByPSO'] = clusteringInfo['ClustersSelectedByPSO']
    results['TimeForPSO1'] = clusteringInfo['TimeForPSO1']
    #results['AccAfterStage1'] = clusteringInfo['AccAfterStage1']
    results['total_Classifiers_Count'] = len(classifiers)
    results['selected_Classifiers_Count'] = len(selectedClassifiers)
    results['selected_Classifiers'] = [model_dict['name'] for model_dict in selectedClassifiers]
    results['TimeForPSO2'] = TimeForPSO2
    results['nonOptimized_Accuracy'] = np.mean(nonOptimized_Accuracy)
    results['nonOptimized_stdDEV'] = np.std(nonOptimized_Accuracy)
    results['optimized_Accuracy'] = np.mean(optimized_Accuracy)
    results['optimized_stdDEV'] = np.std(optimized_Accuracy)
    
    return results
