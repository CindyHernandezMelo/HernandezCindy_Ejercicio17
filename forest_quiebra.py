import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from scipy.io import arff

data =  pd.read_csv('datos.csv')
data = data.replace('?', np.nan)
data = data.dropna()

predictors = list(data.keys())
predictors.remove('Target')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    data[predictors], data['Target'], test_size=0.5)

X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
                                    X_test, y_test, test_size=0.6)


clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')

n_trees = np.arange(1,1000,25)
f1_test = []
f1_train = []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test), average='macro'))
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train),average='macro'))
    feature_importance[i, :] = clf.feature_importances_

plt.figure()
plt.scatter(n_trees, f1_test)
plt.scatter(n_trees, f1_train)

arg_max_F1_test = np.argmax(f1_test)

plt.figure(figsize = (20,5))
avg_importance = feature_importance[arg_max_F1_test,:]
a = pd.Series(avg_importance, index=predictors)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')

n_arboles = n_trees[arg_max_F1_test]
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_arboles, max_features='sqrt')
clf.fit(X_train, y_train)
f1_val_1 = sklearn.metrics.f1_score(y_val, clf.predict(X_val), average='macro')
plt.title('Numero de arboles = %i, F1 score = %f'%(n_arboles, (f1_val_1)/2))

plt.savefig('features.png')
