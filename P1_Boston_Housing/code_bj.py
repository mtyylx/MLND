import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv('bj_housing.csv')
prices = data['Value']
features = data.drop('Value', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.8)


# Train the model, and using GridSearch + K-Fold CV for model parameter tuning.
def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth': np.arange(1, 11)}
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid.fit(X, y)
    return grid.best_estimator_


def robust_test(features, prices, models, iteration):
    depth = []
    score = []
    for x in range(iteration):
        X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.8, random_state=x)
        model = fit_model(X_train, y_train)
        depth.append(model.get_params()['max_depth'])
        y_pred = model.predict(pd.DataFrame(X_test))
        score.append(r2_score(y_pred, y_test))
    return depth, score

(depth, score) = robust_test(features, prices, fit_model, 10)
print pd.Series(depth).describe()
print pd.Series(score).describe()

