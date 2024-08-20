from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def trainClassifiers(X, y, params):
    classifiers = []
    index = 0
    
    for learner in params['classifiers']:
        try:
            if learner == 'KNN':
                model = KNeighborsClassifier(n_neighbors=params['n_neighbors']).fit(X, y)
                classifiers.append({'name': 'KNN', 'model': model})
                index += 1
            elif learner == 'SVM':
                model = SVC(kernel='linear').fit(X, y)
                classifiers.append({'name': 'SVM', 'model': model})
                index += 1
            elif learner == 'NB':
                model = GaussianNB().fit(X, y)
                classifiers.append({'name': 'NB', 'model': model})
                index += 1
            elif learner == 'DISCR':
                model = LinearDiscriminantAnalysis().fit(X, y)
                classifiers.append({'name': 'DISCR', 'model': model})
                index += 1
            elif learner == 'DT':
                model = DecisionTreeClassifier().fit(X, y)
                classifiers.append({'name': 'DT', 'model': model})
                index += 1
            elif learner == 'ANN':
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000).fit(X, y)
                classifiers.append({'name': 'ANN', 'model': model})
                index += 1
            else:
                print('Unknown Classifier')
        except Exception as exc:
            print(f'Something happened in trainClassifiers: {exc}')
    
    return classifiers
