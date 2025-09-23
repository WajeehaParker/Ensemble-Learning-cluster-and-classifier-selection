import numpy as np
from scipy.stats import mode
from pyswarm import pso
from trainClassifiers import trainClassifiers
from helpers import compute_weights, weighted_voting
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pyswarms.discrete.binary import BinaryPSO
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

def classifierSelectionPSO2(classifierList, testData, pso_options=None, 
                            n_particles=50, iterations=100):
    try:
        allPredictions = psoPredict(classifierList, testData)
        y_true = testData[:, -1]
        n_classifiers = allPredictions.shape[1]

        # Define the objective function
        def objective_function(swarm):
            n_particles = swarm.shape[0]
            costs = np.zeros(n_particles)
            for i in range(n_particles):
                selected = swarm[i, :].astype(bool)
                if np.sum(selected) == 0:
                    costs[i] = 1.0  # Penalize if no classifiers are selected
                    continue
                predictions_subset = allPredictions[:, selected]

                # compute majority voting
                # unique_classes = np.unique(predictions_subset)
                # if len(unique_classes) == 0:
                #     majority_vote = np.zeros(predictions_subset.shape[0], dtype=int)
                # else:
                #     # Reshape for broadcasting and compute counts
                #     matches = (predictions_subset[:, :, np.newaxis] == unique_classes)
                #     counts = matches.sum(axis=1)
                #     max_indices = counts.argmax(axis=1)
                #     majority_vote = unique_classes[max_indices]
                
                # compute weighted voting
                weights = compute_weights(predictions_subset, y_true)
                majority_vote = weighted_voting(predictions_subset, weights, y_true)

                accuracy = np.mean(majority_vote == y_true)
                costs[i] = 1.0 - accuracy  # Minimize 1 - accuracy
            return costs

        # Set default PSO options if none provided
        pso_options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 5, 'p': 1}

        # Create initial positions (one particle selects all classifiers)
        init_pos = np.zeros((n_particles, n_classifiers), dtype=int)
        init_pos[0, :] = 1  # First particle selects all classifiers

        # make second particle select best solution from previous run
        # make every 5th position in init_pos[1, :] as one
        init_pos[1, :] = (np.arange(n_classifiers) % 5 == 0).astype(int)

        for i in range(2, n_particles):
            init_pos[i] = np.random.randint(2, size=n_classifiers)

        # Initialize BinaryPSO with one particle starting with all classifiers selected
        optimizer = BinaryPSO(
            n_particles=n_particles,
            dimensions=n_classifiers,
            options=pso_options,
            init_pos=init_pos
        )

        # Run optimization
        cost, pos = optimizer.optimize(
            objective_function,
            iters=iterations,
            # init_pos=init_pos
        )

        # Get best combination
        selected_indices = np.where(pos)[0].tolist()
        if not selected_indices:  # Fallback if no classifiers selected
            selected_indices = [0]
        # best_predictions = allPredictions[:, selected_indices]
        # majority_vote = scipy.stats.mode(best_predictions, axis=1, keepdims=False).mode
        # best_accuracy = np.mean(majority_vote == y_true)
        # print(f'Best accuracy: {best_accuracy:.4f}')

        return selected_indices

    except Exception as exc:
        print(f'Problem in classifierSelectionPSO2: {exc}')
        return []

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
    #psoEnsemble = classifierSelectionPSO(classifiers, np.column_stack((valX, valy)))
    psoEnsemble = classifierSelectionPSO2(classifiers, np.column_stack((valX, valy)))
    #classifiers, selectedClassifiers, meta_accuracy = classifierSelectionWithSHAP(selectedClusters, valX, valy, params)
    #print("Classifier SVM duration: ", TimeForPSO2)
    #psoEnsemble = np.flatnonzero(psoEnsemble['chromosome'])
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