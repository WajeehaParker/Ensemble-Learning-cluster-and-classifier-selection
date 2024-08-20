import numpy as np
from scipy.stats import mode
from pyswarm import pso
from trainClassifiers import trainClassifiers

def psoPredict(classifiers, testData):
    X = testData[:, :-1]
    predictions = np.ones((len(testData[:, -1]), len(classifiers)))
    
    for i in range(len(classifiers)):
        try:
            predictions[:, i] = classifiers[i].model.predict(X)
        except Exception as ME:
            print(f'IN psoPredict: {str(ME)}')
            continue
    
    return predictions

def classifierSelectionPSO(classifierList, testData):
    def PSOAF(c):
        c = c > 0.6
        c = np.where(c)[0]
        
        decisionMatrix = np.ones((len(testData[:, -1]), len(c)))
        for i in range(len(c)):
            decisionMatrix[:, i] = allPredictions[:, c[i]]
        
        decisionMatrix = mode(decisionMatrix, axis=1)[0]
        error = np.mean(decisionMatrix != testData[:, -1])
        return error

    try:
        allPredictions = psoPredict(classifierList, testData)
        lb = np.zeros(len(classifierList))
        ub = np.ones(len(classifierList))
        best, fval = pso(PSOAF, lb, ub, swarmsize=50)
        obj = {
            'chromosome': np.round(best),
            'fval': fval
        }
    except Exception as exc:
        print(f'Problem with {exc}')
        obj = None

    return obj

def classifierSelection(selectedClusters, valX, valy, params):
    classifiers = []
    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        all = trainClassifiers(X, y, params)
        classifiers.extend(all)

    psoEnsemble = classifierSelectionPSO(classifiers, np.column_stack((valX, valy)))
    psoEnsemble = np.flatnonzero(psoEnsemble['chromosome'])
    selectedClassifiers = [classifiers[i] for i in psoEnsemble]
    return classifiers, selectedClassifiers