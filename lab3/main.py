import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv('howpop_train.csv')
data.drop(filter(lambda c: c.endswith('_lognorm'), data.columns),
          axis=1, inplace=True)

sn.set_style('dark')
sn.set_palette('RdBu')
sn.set_context('notebook', font_scale=1.5,
               rc={'figure.figsize': (15, 5),
                   'axes.titlesize': 18})

data['published'] = pd.to_datetime(data.published, yearfirst = True)

data['year'] = [d.year for d in data.published]
data['month'] = [d.month for d in data.published]
data['dayofweek'] = [d.isoweekday() for d in data.published]
data['hour'] = [d.hour for d in data.published]

monthly_stats = data.groupby(['year', 'month'])['url'].count().reset_index()
most_popular_month = monthly_stats.loc[monthly_stats['url'].idxmax()]
plt.figure(figsize=(12, 6))
sn.barplot(x='month', y='url', hue='year', data=monthly_stats)
plt.title(f'Most popular month: {most_popular_month["month"]}/{most_popular_month["year"]}')
plt.xlabel('Month')
plt.ylabel('Number of publications')
plt.show()
print(f'березень 2015')

selected_month = most_popular_month['month']
selected_year = most_popular_month['year']
selected_data = data[(data['month'] == selected_month) & (data['year'] == selected_year)]
plt.figure(figsize=(12, 6))
sn.countplot(x='dayofweek', hue='domain', data=selected_data)
plt.title(f'Daily distribution of publications in {selected_month}/{selected_year}')
plt.xlabel('Day of week')
plt.ylabel('Number of publications')
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend(title='Domain')
plt.show()
print('на хабрі завжди більше статей, ніж на гіктаймсі')
print('по суботам на гіктаймс і на хабрахабр публікують приблизно одинакову кількість статей.')

plt.figure(figsize=(12, 6))
data['hour'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of published articles by hours')
plt.xlabel('Publication hour')
plt.ylabel('Number of articles')
plt.xticks(rotation=0)
plt.show()
print('більш всього переглядів набирають статті, опубліковані в 12 годин дня')

top_authors = data['author'].value_counts().head(20).index
top_authors_data = data[data['author'].isin(top_authors)]
author_minus_avg = top_authors_data.groupby('author')['votes_minus'].mean()
plt.figure(figsize=(12, 6))
author_minus_avg.sort_values(ascending=False).plot(kind='bar', color='salmon')
plt.title('The average number of negatives for the top 20 authors')
plt.xlabel('Author')
plt.ylabel('The average number of negatives')
plt.xticks(rotation=45)
plt.show()
print('@Mithgol')

saturdays = data[data['dayofweek'] == 6]
mondays = data[data['dayofweek'] == 1]
saturday_hours = saturdays['hour'].value_counts().sort_index()
monday_hours = mondays['hour'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(saturday_hours.index, saturday_hours.values, label='Saturday', marker='o', linestyle='-')
plt.plot(monday_hours.index, monday_hours.values, label='Monday', marker='o', linestyle='-')
plt.title('Distribution of publication hours on Saturdays and Mondays')
plt.xlabel('Hour')
plt.ylabel('Number of publications')
plt.xticks(range(24))
plt.legend()
plt.grid(True)
plt.show()