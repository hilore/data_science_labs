import pandas as pd
from utils import *

data = pd.read_csv('./data/adult.data.csv', delimiter=',')
print(data.head())
print()

print('==== Task 1 ====')
print(data['sex'].value_counts(), '\n')

print('==== Task 2 ====')
female_data = data[data['sex'] == 'Female']
print(f'{female_data["age"].mean():.2f}\n')

print('==== Task 3 ====')
total_people = data.shape[0]
german_people = data[data['native-country'] == 'Germany'].shape[0]
german_people_percent_result = (german_people / total_people) * 100
print(f'{german_people_percent_result:.2f}%\n')

print('==== Task 4-5 ====')
high_income = data[data['salary'] == '>50K']
low_income = data[data['salary'] == '<=50K']

avg_age_high_income = high_income['age'].mean()
std_dev_age_high_income = high_income['age'].std()

avg_age_low_income = low_income['age'].mean()
std_dev_age_low_income = low_income['age'].std()
print(f'Average age for those earning over 50K: {avg_age_high_income:.2f} years')
print(f'Average age deviation for those earning over 50K: {std_dev_age_high_income:.2f} years')
print(f'Average age for those earning less than 50K: {avg_age_low_income:.2f} years')
print(f'Average age deviation for those earning less than 50K: {std_dev_age_low_income:.2f} years\n')

print('==== Task 6 ====')
find_not_high_education = False
high_education = ('Bachelors', 'Prof-school', 'Assoc-acdm',
					'Assoc-voc', 'Masters', 'Doctorate')

for education_item in high_income['education']:
	if education_item not in high_education:
		find_not_high_education = True
		break

if find_not_high_education:
	print(f'There are people with incomes over 50K who don\'t have a college degree\n')
else:
	print('All people with income over 50K have a college degree\n')

print('==== Task 7 ====')
race_sex_age_stats = data.groupby(['race', 'sex'])['age'].describe()
max_age_amind_eskimo_male = race_sex_age_stats.loc[('Amer-Indian-Eskimo', 'Male')]['max']
print(f'Age statistics for race and sex:\n{race_sex_age_stats}\n')
print(f'Maximum age for men of the Amer-Indian-Eskimo race: {max_age_amind_eskimo_male}\n')

print('==== Task 8 ====')
data['marital-status-category'] = data.apply(get_marital_status, axis=1)
income_group = data[data['salary'] == '>50K']
income_distribution = income_group.groupby('marital-status-category')['salary'].count() / data.groupby('marital-status-category')['salary'].count()
print(income_distribution,'\n')
print('%')
print(income_distribution * 100,'\n')

print('==== Task 9 ====')
max_hours_per_week = data['hours-per-week'].max()
max_hours_workers = data[data['hours-per-week'] == max_hours_per_week]
max_hours_workers_cnt = max_hours_workers.shape[0]
high_income_max_hours_percentage = (max_hours_workers[max_hours_workers['salary'] == '>50K'].shape[0] / max_hours_workers_cnt) * 100
print(f'Maximum number of hours worked per week: {max_hours_per_week}')
print(f'Number of people working {max_hours_per_week} hours per week: {max_hours_workers_cnt}')
print(f'Percentage of workers working {max_hours_per_week} hours and earning over 50K: {high_income_max_hours_percentage:.2f}%\n')

print('==== Task 10 ====')
avg_hours_per_week_by_country = data.groupby(['native-country', 'salary'])['hours-per-week'].mean()
print(avg_hours_per_week_by_country)