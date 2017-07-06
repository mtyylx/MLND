import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)    # features = data[['RM', 'LSTAT', 'PTRATIO']]
X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.8, random_state=0)


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth': np.arange(1, 11)}
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid.fit(X, y)
    return grid.best_estimator_


model = fit_model(X_train, y_train)
print model.get_params()['max_depth']

print model.predict([[5, 17, 15]])

client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

from sklearn.neighbors import NearestNeighbors

nb = NearestNeighbors(10)
nb.fit(X_train)
print nb.kneighbors(client_data, 1)
print X_train.iloc[[101]]