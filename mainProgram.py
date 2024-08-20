import os
import csv
from runTraining import runTraining
import warnings

warnings.filterwarnings("ignore")

def saveResults(results):
    file_path = 'results.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Data Set', 'Classifier', 'Avg Accuracy', 'Std. Dev', 'Optimized Acc', 'Std. Dev', 'Duration'])

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            results['p_name'],
            results['selected_Classifiers'],
            results['nonOptimized_Accuracy'],
            results['nonOptimized_stdDEV'],
            results['optimized_Accuracy'],
            results['optimized_stdDEV'],
        ])

def run_problem(p_name, params):
    results = {}
    print("p_name: "+p_name)
    results = runTraining(p_name, params)
    saveResults(results)

def mainProgram():
    problem = [
                 'breast-cancer-wisconsin', 
                 'ecoli',
                 'haberman', 
                 'ionosphere', 
                 'iris', 
                 'liver',
                 'pima_diabetec',
                 'sonar', 
                 'wine',
                # 'forest-cover',
                # 'german-credit',
                # 'adult-income'
                ]
    #'diabetic_retinopathy', 'segment2', 'thyroid', 'vehicle'

    params = {
        'classifiers': ['DT', 'ANN', 'KNN', 'DISCR', 'NB'], #'SVM'
        'n_neighbors': 1    #for KNN
    }
    
    for p_name in problem:
        run_problem(p_name, params)

num_run = 10
for i in range(num_run):
    mainProgram()