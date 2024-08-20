import os
import numpy as np
from clusterSelection import clusterSelection
from classifierSelection import classifierSelection
from scipy.stats import mode
from sklearn.model_selection import train_test_split

def readData(p_name):
    file_path = os.path.join('DTE', p_name, 'data.csv')
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Adjust delimiter if necessary
    X = data[:, :-1]
    Y = data[:, -1]
    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # Adjust test_size if needed
    trainX, valX, trainy, valy = train_test_split(X, Y, test_size=0.25, random_state=42)  # Adjust test_size if needed
    return trainX, trainy, valX, valy, X_test, y_test

def fusion(classifiers, data):
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
    return acc

def runTraining(p_name, params):
    results = {}
    nonOptimized_Accuracy = []
    optimized_Accuracy = []
    
    trainX, trainy, valX, valy, X_test, y_test = readData(p_name)
    print("Data Loaded Successfully")    

    selectedClusters = clusterSelection(trainX, trainy, valX, valy, params)
    classifiers, selectedClassifiers = classifierSelection(selectedClusters, valX, valy, params)

    nonOptimized_Accuracy.append(fusion(classifiers, np.column_stack((X_test, y_test))))
    optimized_Accuracy.append(fusion(selectedClassifiers, np.column_stack((X_test, y_test))))
    #end for

    results['p_name'] = p_name
    results['selected_Classifiers'] = [model_dict['name'] for model_dict in selectedClassifiers]
    #results['selected_Classifiers'] = selectedClassifiers
    results['nonOptimized_Accuracy'] = np.mean(nonOptimized_Accuracy)
    results['nonOptimized_stdDEV'] = np.std(nonOptimized_Accuracy)
    results['optimized_Accuracy'] = np.mean(optimized_Accuracy)
    results['optimized_stdDEV'] = np.std(optimized_Accuracy)
    
    return results
