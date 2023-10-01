import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

print(data.head(4))
print(data.describe(include='all'))
print(data.info())

data = data.drop('Cabin', axis=1).dropna()

print('==== Task 4.1 ====')
selected_cols = ['Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Survived']
sn.pairplot(data[selected_cols], hue='Survived', palette={0: "red", 1: "blue"})
plt.show()

print('\n==== Task 4.2 ====')
pclass_fare_data = data[['Pclass', 'Fare']]
plt.figure(figsize=(8, 6))
sn.boxplot(x='Pclass', y='Fare', data=pclass_fare_data)
plt.title('Dependence of Fare on Pclass')
plt.show()

print('\n==== Task 5(8) ====')
plt.figure(figsize=(8,6))
sn.countplot(x='Sex', hue='Survived', data=data)
plt.title('The ratio of dead and survivors depending on gender')
plt.xlabel('Sex')
plt.ylabel('Quantity')
plt.show()

print('\n==== Task 6(9) ====')
plt.figure(figsize=(8,6))
sn.countplot(x='Pclass', hue='Survived', data=data)
plt.title('The ratio of dead and survivors depending on pclass')
plt.xlabel('Pclass')
plt.ylabel('Quantity')
plt.show()

print('\n ==== Task 7(10) ====')
data['AgeCategory'] = 'Young'
data.loc[data['Age'] > 60, 'AgeCategory'] = 'Old'
plt.figure(figsize=(8, 6))
sn.countplot(x='AgeCategory', hue='Survived', data=data)
plt.title('The ratio of survival to age')
plt.xlabel('Age category')
plt.ylabel('Quantity')
plt.show()
