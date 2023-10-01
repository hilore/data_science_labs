import collections
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('adult_train.csv', sep=';')
print(data_train.tail())

data_test = pd.read_csv('adult_test.csv', sep=';')
data_test = data_test[data_test['Age'] != '|1x3 Cross validator']
print(data_test.tail())

data_train.loc[data_train['Target'] == ' <=50K', 'Target'] = 0
data_train.loc[data_train['Target'] == ' >50K', 'Target'] = 1
data_test.loc[data_test['Target'] == ' <=50K.', 'Target'] = 0
data_test.loc[data_test['Target'] == ' >50K.', 'Target'] = 1

print(data_test.describe(include='all').T)
print(data_train['Target'].value_counts())

fig = plt.figure(figsize=(25, 15))
cols = 5
rows = int(np.ceil(float(data_train.shape[1]) / cols))

for i, col in enumerate(data_train.columns, start=1):
    ax = fig.add_subplot(rows, cols, i)
    ax.set_title(col)
    if data_train.dtypes[col] == np.object_:
        data_train[col].value_counts().plot(kind='bar', ax=ax)
    else:
        data_train[col].hist(ax=ax)
        plt.xticks(rotation='vertical')
    
plt.subplots_adjust(hspace=0.7, wspace=0.2)

data_test['Age'] = data_test['Age'].astype(int)
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)
data_test['Education_Num'] = data_test['Education_Num'].astype(int)
data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)
data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)
data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)

categorical_columns_train = [c for c in data_train.columns if data_train[c].dtype.name == 'object']
numerical_columns_train = [c for c in data_train.columns if data_train[c].dtype.name != 'object']
categorical_columns_test = [c for c in data_test.columns if data_test[c].dtype.name == 'object']
numerical_columns_test = [c for c in data_test.columns if data_test[c].dtype.name != 'object']

print(f'categorical_columns_train : {categorical_columns_train}')
print(f'numerical_columns_train : {numerical_columns_train}')
print(f'categorical_columns_test : {categorical_columns_test}')
print(f'numerical_columns_test : {numerical_columns_test}')

for col in categorical_columns_train:
    data_train[col] = data_train[col].fillna(data_train[col].mode())

for col in categorical_columns_test:
    data_test[col] = data_test[col].fillna(data_train[col].mode())

for col in numerical_columns_train:
    data_train[col] = data_train[col].fillna(data_train[col].mode())

for col in numerical_columns_test:
    data_test[col] = data_test[col].fillna(data_train[col].mode())

data_train = pd.concat([data_train, pd.get_dummies(data_train['Workclass'], prefix='Workclass'), pd.get_dummies(data_train['Education'], prefix='Education'), pd.get_dummies(data_train['Martial_Status'], prefix='Martial_Status'), pd.get_dummies(data_train['Occupation'], prefix='Occupation'), pd.get_dummies(data_train['Relationship'], prefix='Relationship'), pd.get_dummies(data_train['Race'], prefix='Race'), pd.get_dummies(data_train['Sex'], prefix='Sex'), pd.get_dummies(data_train['Country'], prefix='Country')], axis=1)

data_test = pd.concat([data_test, pd.get_dummies(data_test['Workclass'], prefix='Workclass'), pd.get_dummies(data_test['Education'], prefix='Education'), pd.get_dummies(data_test['Martial_Status'], prefix='Martial_Status'), pd.get_dummies(data_test['Occupation'], prefix='Occupation'), pd.get_dummies(data_test['Relationship'], prefix='Relationship'), pd.get_dummies(data_test['Race'], prefix='Race'), pd.get_dummies(data_test['Sex'], prefix='Sex'), pd.get_dummies(data_test['Country'], prefix='Country')], axis=1)

data_train.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'], axis=1, inplace=True)
data_test.drop(['Workclass', 'Education', 'Martial_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country'], axis=1, inplace=True)

print(data_test.describe(include='all').T)
print(set(data_train.columns) - set(data_test.columns))

data_test['Country_ Holand-Netherlands'] = np.zeros([data_test.shape[0], 1])
print(set(data_train.columns) - set(data_test.columns))
print(data_train.head(2))
print(data_test.head(2))

train_cols_order = data_train.columns
data_test = data_test[train_cols_order]

X_train=data_train.drop(['Target'], axis=1)
y_train = data_train['Target']
X_test=data_test.drop(['Target'], axis=1)
y_test = data_test['Target']

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)


print('\n==== Task 12 ====')
tree_predictions = tree.predict(X_test)
accuracy = accuracy_score(y_test, tree_predictions)
print(f'Accuracy of predictions on test data: {accuracy * 100:.2f}%\n')

print('==== Task 13 ====')
tree_params = {'max_depth': range(2,11)}
locally_best_tree = GridSearchCV(tree, tree_params, cv=5)
locally_best_tree.fit(X_train, y_train)
print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

print('\n==== Task 14 ====')
tuned_tree = DecisionTreeClassifier(max_depth=9, random_state=17)
tuned_tree.fit(X_train, y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
accuracy = accuracy_score(y_test, tuned_tree_predictions)
print(f'Accuracy of predictions on test data: {accuracy * 100:.2f}%\n')

print('==== Task 15 ====')
rf = RandomForestClassifier(n_estimators=100, random_state=17)
rf.fit(X_train, y_train)

print(f'\n====Task 16 ====')
forest_predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, forest_predictions)
print(f'Accuracy of Random Forest predictions on test data: {accuracy * 100:.2f}%\n')

print('==== Task 17 ====')
forest_params = {'max_depth': range(10, 21),
                 'max_features': range(5, 105, 10)}
locally_best_forest = GridSearchCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=17),
    param_grid=forest_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
locally_best_forest.fit(X_train, y_train)
print("Best params:", locally_best_forest.best_params_)
print("Best cross validaton score", locally_best_forest.best_score_)

print('\n==== Task 18 ====')
tuned_forest_predictions = locally_best_forest.predict(X_test)
accuracy = accuracy_score(y_test, tuned_forest_predictions)
print(f'Accuracy of predictions on test data: {accuracy * 100:.2f}%')