import os
import csv
from runTraining import runTraining
import warnings
import time

warnings.filterwarnings("ignore")

def saveResults(results, duration):
    file_path = 'results.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Data Set', 'Total Clusters Count', 'Non Homogenous Clusters Count', 'Clusters Selected by PSO', 'Time For PSO 1', 'Total Classifiers Count', 'Selected Classifiers Count', 'Classifier', 'Time For PSO 2', 'Avg Accuracy', 'Std. Dev', 'Optimized Acc', 'Std. Dev', 'Duration'])

    minutes = int(duration // 60) 
    seconds = int(duration % 60) 

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            results['p_name'],
            results['TotalClustersCount'],
            results['NonHomogenousClustersCount'],
            results['ClustersSelectedByPSO'],
            results['TimeForPSO1'],
            results['total_Classifiers_Count'],
            results['selected_Classifiers_Count'],
            results['selected_Classifiers'],
            results['TimeForPSO2'],
            results['nonOptimized_Accuracy'],
            results['nonOptimized_stdDEV'],
            results['optimized_Accuracy'],
            results['optimized_stdDEV'],
            f"{minutes}m {seconds}s"
        ])

def run_problem(p_name, params):
    results = {}
    print("p_name: "+p_name)
    start_time = time.time()
    results = runTraining(p_name, params)
    end_time = time.time()
    duration = end_time - start_time
    saveResults(results, duration)

def mainProgram():
    problem = [
                # 'breast-cancer-wisconsin', 
                # 'ecoli',
                # 'haberman', 
                # 'ionosphere', 
                # 'iris', 
                # 'liver',
                # 'pima_diabetec',
                # 'sonar', 
                # 'wine',
                # 'forest-cover',
                # 'german-credit',
                # 'adult-income',
                 'credit-card-default',
                 'bank-marketing'
                ]
    #'diabetic_retinopathy', 'segment2', 'thyroid', 'vehicle'

    params = {
        #'classifiers': ['SVM'],
        'classifiers': ['DT', 'ANN', 'KNN', 'DISCR', 'NB'], #'SVM'
        'n_neighbors': 1    #for KNN
    }
    
    for p_name in problem:
        run_problem(p_name, params)

num_run = 1
for i in range(num_run):
    mainProgram()