import numpy as np
from scipy.stats import mode
from pyswarm import pso
from trainClassifiers import trainClassifiers
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import time

def psoPredict(classifiers, testData):
    X = testData[:, :-1]
    predictions = np.ones((len(testData[:, -1]), len(classifiers)))
    
    for i in range(len(classifiers)):
        try:
            predictions[:, i] = classifiers[i].model.predict(X)
        except Exception as ME:
            #print(f'IN psoPredict: {str(ME)}')
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

def classifierSelectionPSOwithDiversity(classifierList, testData):
    def PSOAF(c):
        c = c > 0.6
        c = np.where(c)[0]
        
        # Step 1: Accuracy Calculation
        decisionMatrix = np.ones((len(testData[:, -1]), len(c)))
        for i in range(len(c)):
            decisionMatrix[:, i] = allPredictions[:, c[i]]
        
        decisionMatrix = mode(decisionMatrix, axis=1)[0]
        error = np.mean(decisionMatrix != testData[:, -1])

        ########## change ##########
        # Step 2: Diversity Calculation
        diversity_sum = 0
        num_pairs = 0
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                pred_i = allPredictions[:, c[i]]
                pred_j = allPredictions[:, c[j]]

                N11 = np.sum((pred_i == 1) & (pred_j == 1))
                N00 = np.sum((pred_i == 0) & (pred_j == 0))
                N10 = np.sum((pred_i == 1) & (pred_j == 0))
                N01 = np.sum((pred_i == 0) & (pred_j == 1))
                total = N11 + N10 + N01 + N00
                
                if total > 0:  # Avoid division by zero
                    DF = N00 / total
                    diversity_sum += DF
                    num_pairs += 1

        # Average diversity across all pairs
        diversity = diversity_sum / num_pairs if num_pairs > 0 else 0

        # Step 3: Combine accuracy and diversity into the fitness function
        alpha = 0.7  # Weight for accuracy
        beta = 0.3   # Weight for diversity
        fitness = alpha * error - beta * diversity  # Minimize error, maximize diversity
        return fitness
        ########## change end ##########
        #return error

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
    start_time = time.time()
    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        all = trainClassifiers(X, y, params)
        classifiers.extend(all)

    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60) 
    seconds = int(duration % 60) 
    TimeForPSO2 = f"{minutes}m {seconds}s"
    psoEnsemble = classifierSelectionPSO(classifiers, np.column_stack((valX, valy)))
    #print("Classifier SVM duration: ", TimeForPSO2)
    psoEnsemble = np.flatnonzero(psoEnsemble['chromosome'])
    selectedClassifiers = [classifiers[i] for i in psoEnsemble]
    return classifiers, selectedClassifiers, TimeForPSO2

def classifierSelectionWithSHAP(selectedClusters, valX, valy, params):
    classifiers = []
    meta_train_X = []
    meta_train_y = []

    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        trained_classifiers = trainClassifiers(X, y, params) 
        classifiers.extend(trained_classifiers)

    for clf in classifiers:
        predictions = clf['model'].predict(valX)
        meta_train_X.append(predictions)

    meta_train_X = np.array(meta_train_X).T
    meta_train_y = valy

    print("Training Meta Learner at:", datetime.now())
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(meta_train_X, meta_train_y)

    meta_predictions = meta_learner.predict(meta_train_X)
    meta_accuracy = accuracy_score(meta_train_y, meta_predictions)
    print(f"Meta Learner Accuracy: {meta_accuracy:.4f}")

    print("Calculating Classifier Importance with SHAP at:", datetime.now())
    shap_explainer = shap.Explainer(meta_learner, meta_train_X)
    shap_values = shap_explainer(meta_train_X)
    shap_mean_importances = np.abs(shap_values.values).mean(axis=0) 

    importance_threshold = np.percentile(shap_mean_importances, 75)  
    selected_indices = np.where(shap_mean_importances >= importance_threshold)[0]
    selectedClassifiers = [classifiers[i] for i in selected_indices]

    return classifiers, selectedClassifiers, meta_accuracy