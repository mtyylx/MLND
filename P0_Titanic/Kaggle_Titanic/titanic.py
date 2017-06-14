import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, make_scorer

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def preprocess(f):
    f = f[['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare', 'Embarked']].copy()

    # Preprocess 'Pclass'
    # f.loc[f.Pclass == 1, 'Pclass'] = 'a'
    # f.loc[f.Pclass == 2, 'Pclass'] = 'b'
    # f.loc[f.Pclass == 3, 'Pclass'] = 'c'

    # Preprocess 'Sex'
    f.loc[f.Sex == 'male', 'Sex'] = 1
    f.loc[f.Sex == 'female', 'Sex'] = 0
    f.Sex = f.Sex.astype(int)

    # Preprocess 'Age'
    f.loc[np.isnan(f.Age), 'Age'] = -1
    mean = f.Age.mean()
    std = f.Age.std()
    f.loc[:, 'Age'] = (f.loc[:, 'Age'] - mean) / std

    # Preprocess 'Fare'
    f.loc[np.isnan(f.Fare), 'Fare'] = 0
    mean = f.Fare.mean()
    std = f.Fare.std()
    f.loc[:, 'Fare'] = (f.loc[:, 'Fare'] - mean) / std

    f = pd.get_dummies(f)

    return f


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # clf = DecisionTreeClassifier(random_state=0)
    clf = RandomForestClassifier(random_state=0)
    params = {'n_estimators': np.arange(300, 400, 100)}
    scorer = make_scorer(r2_score)
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer, cv=cv_sets)
    grid.fit(X, y)
    print 'Best Depth =', grid.best_estimator_.get_params()['n_estimators']
    return grid.best_estimator_


f = preprocess(train)
print list(f.columns)
survived = train.Survived.copy()
X_train, X_test, y_train, y_test = train_test_split(f, survived, random_state=0)
clf = fit_model(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
print r2_score(y_test, y_pred)
print clf.feature_importances_

f = preprocess(test)
result = clf.predict(f)
result = pd.DataFrame({'PassengerId': pd.Series(test.PassengerId), 'survived': pd.Series(result)}, columns=['PassengerId', 'survived'])
result.to_csv('output.csv', index=False)
