import numpy as np
import pandas as pd

def get_gender(female) -> str:
    return 'female' if female else 'male'

def main():
    data_folder = './data'
    np.set_printoptions(precision=2)

    beauty_data = pd.read_csv(data_folder + '/beauty.csv', delimiter=';')
    print('First 5 rows:')
    print(beauty_data.head(), end='\n\n')

    print(f'Shape: {beauty_data.shape}\n')
    
    print(f'Describe:{beauty_data.describe()}\n')
    print(f'Indexing:\n{beauty_data["exper"].head()}\n')

    print(f'loc/iloc:')
    print(beauty_data.loc[:5, ['wage', 'female']])
    print(beauty_data.iloc[:,2:4].head())
    
    logic_indexing = (beauty_data[beauty_data['female'] == 1]['wage'].mean(),
                      beauty_data[beauty_data['female'] == 0]['wage'].mean())
    print('\nLogic indexing:', logic_indexing, end='\n\n')

    print(f'Groupby:')
    for look, sub_df in beauty_data.groupby('looks'):
        print(look)
        print(sub_df['goodhlth'].mean())
    print(beauty_data.groupby('looks')[['wage', 'exper']].agg(np.median))

    print('\nTable:')
    print(pd.crosstab(beauty_data['female'], beauty_data['married']), end='\n\n')
    print(pd.crosstab(beauty_data['female'], beauty_data['looks']))

    print(f'\nAdding columns:')
    beauty_data['is_rich'] = (beauty_data['wage'] > beauty_data['wage'].quantile(.75)).astype('int64')
    print(beauty_data.head())
    beauty_data['rubbish'] = .56 * beauty_data['wage'] + 0.32 * beauty_data['exper']
    print(beauty_data.head(), end='\n\n')

    print('map/apply:')
    d = {1: 'union', 0: 'non-union'}
    print(beauty_data['union'].map(d).head())
    print(beauty_data['female'].apply(lambda female: get_gender(female)).head())

if __name__ == '__main__':
    main()
