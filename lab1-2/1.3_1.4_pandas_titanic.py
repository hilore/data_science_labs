import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/titanic_train.csv', index_col='PassengerId')
print(data.head(4))

print('==== Task 1 ====')
print(data.groupby('Sex')['Survived'].count())

print('==== Task 2 ====')
total_pclass = data.groupby('Pclass')['Survived']
sex_pclass = data.groupby(['Pclass', 'Sex'])['Survived']
male_pclass_2 = data[(data['Pclass'] == 2) & (data['Sex'] == 'male')].shape[0]
print(f'{total_pclass.count()}\n')
print(f'{sex_pclass.count()}\n')
print(f'Male Pclass 2:\n{male_pclass_2}\n')

print('==== Task 3 ====')
fare_median = round(data['Fare'].median(), 2)
fare_std = round(data['Fare'].std(), 2)
print(f'Median Fare: {fare_median}')
print(f'Std Fare: {fare_std}\n')

print('==== Task 4 ====')
yound_survived = round(data[data['Age'] < 30]['Survived'].mean() * 100, 1)
old_survived = round(data[data['Age'] > 60]['Survived'].mean() * 100, 1)
print(f'Young - {yound_survived}, old - {old_survived}\n')

print('==== Task 5 ====')
male_survived = round(data[data['Sex'] == 'male']['Survived'].mean() * 100, 1)
female_survived = round(data[data['Sex'] == 'female']['Survived'].mean() * 100, 1)
print(f'Male - {male_survived}, female - {female_survived}\n')

print('==== Task 6 ====')
male_pgrs = data[data['Sex'] == 'male'].copy()
male_pgrs['FirstName'] = male_pgrs['Name'].apply(lambda x: x.split(', ')[1].split('. ')[1].split(' ')[0])
print(f'The most popular name among men: {male_pgrs["FirstName"].value_counts().head(1)}')

print('==== Task 7 ====')
survived = data[data['Survived'] == 1]
not_survived = data[data['Survived'] == 0]

# розподіл вартості квитків у врятованих та загиблих
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(survived['Fare'], bins=20, alpha=0.5, label='Survived', color='green')
plt.hist(not_survived['Fare'], bins=20, alpha=0.5, label='Dead', color='red')
plt.xlabel('Fare cost')
plt.ylabel('Passangers number')
plt.legend()

# розподіл віку у врятованих і загиблих
plt.subplot(1, 2, 2)
plt.hist(survived['Age'].dropna(), bins=20, alpha=0.5, label='Survived', color='green')
plt.hist(not_survived['Age'].dropna(), bins=20, alpha=0.5, label='Dead', color='red')
plt.xlabel('Age')
plt.ylabel('Passangers number')
plt.legend()

plt.tight_layout()
plt.show()

average_age_survived = survived['Age'].mean()
average_age_not_survived = not_survived['Age'].mean()
print(f'Avg age for survived = {int(average_age_survived)}')
print(f'Avg age for not survived = {int(average_age_not_survived)}\n')

print('==== Task 8 ====')
male_passengers = data[data['Sex'] == 'male']
female_passengers = data[data['Sex'] == 'female']

male_age_by_class = male_passengers.groupby('Pclass')['Age'].mean()
female_age_by_class = female_passengers.groupby('Pclass')['Age'].mean()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
male_age_by_class.plot(kind='bar', color='blue')
plt.title('Average age of men by service class')
plt.xlabel('Service class')
plt.ylabel('Average age')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
female_age_by_class.plot(kind='bar', color='purple')
plt.title('Average age of men by service class')
plt.xlabel('Service class')
plt.ylabel('Average age')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
