import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def get_auc_lr_valid(X, y, C=1.0, ratio=0.9, seed=17):
    train_len = int(ratio * X.shape[0])
    X_train_part = X[:train_len, :]
    X_valid_part = X[train_len:, :]
    y_train_part = y[:train_len]
    y_valid_part = y[train_len:]

    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear')
    lr.fit(X_train_part, y_train_part)

    y_pred = lr.predict_proba(X_valid_part)[:, 1]

    auc = roc_auc_score(y_valid_part, y_pred)
    return auc

def write_to_submission_file(predict_labels, out_file='submission.csv', target='target', index_label='session_id'):
    predicted_data = pd.DataFrame(predict_labels,
                                  index=np.arange(1, predict_labels.shape[0] + 1),
                                  columns=[target])
    predicted_data.to_csv(out_file, index_label=index_label)

train_data = pd.read_csv('./train_sessions.csv', index_col='session_id')
test_data = pd.read_csv('./test_sessions.csv', index_col='session_id')

times = ['time%s' % i for i in range(1, 11)]
train_data[times] = train_data[times].apply(pd.to_datetime)
test_data[times] = test_data[times].apply(pd.to_datetime)

train_data = train_data.sort_values(by='time1')
print(train_data.head())

sites = ['site%s' % i for i in range(1, 11)]
train_data[sites] = train_data[sites].fillna(0).astype(int)
test_data[sites] = test_data[sites].fillna(0).astype(int)

with open('site_dic.pkl', 'rb') as input_file:
    site_dct = pickle.load(input_file)

sites_dct_data = pd.DataFrame(list(site_dct.keys()),
                              index=list(site_dct.values()),
                              columns=['site'])
print(f'Total sites: {sites_dct_data.shape[0]}')
print(sites_dct_data.head())

y_train = train_data['target']
full_data = pd.concat([train_data.drop('target', axis=1),
                       test_data])
idx_split = train_data.shape[0]

full_sites = full_data[sites]
print(full_sites.head())

sites_flatten = full_sites.values.flatten()
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

X_train, X_valid, y_train, y_valid = train_test_split(full_sites_sparse[:idx_split], y_train,
                                                      test_size=0.1, random_state=17,
                                                      stratify=y_train)

auc = get_auc_lr_valid(X_train, y_train)
print(f'ROC AUC on the validation sample: {auc}')

lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear')
lr.fit(full_sites_sparse[:y_train.shape[0]], y_train)
y_pred_valid = lr.predict_proba(X_valid)[:, 1]
auc_valid = roc_auc_score(y_valid, y_pred_valid)
print(f'ROC AUC on delayed sample: {auc_valid}')

y_pred_test = lr.predict_proba(full_sites_sparse[idx_split:])[:, 1]
write_to_submission_file(y_pred_test)

full_data['time1'] = pd.to_datetime(full_data['time1'])
full_data['year_month'] = full_data['time1'].apply(lambda x: x.strftime('%Y%m'))

scaler = StandardScaler()
full_data['year_month_scaled'] = scaler.fit_transform(full_data['year_month'].values.reshape(-1, 1))

new_feature = full_data['year_month_scaled'].values.reshape(-1, 1)
full_sites_sparse = hstack([full_sites_sparse, new_feature]).tocsr()

lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear')
lr.fit(X_train, y_train)

y_pred_valid = lr.predict_proba(X_valid)[:, 1]
auc_valid = roc_auc_score(y_valid, y_pred_valid)
print(f'ROC AUC on delayed sample with new feature (year_month_scaled): {auc_valid}')

full_data['start_hour'] = full_data['time1'].apply(lambda x: x.hour)
full_data['morning'] = (full_data['start_hour'] <= 11).astype(int)
lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear')
lr.fit(X_train, y_train)
y_pred_valid = lr.predict_proba(X_valid)[:, 1]
auc_valid = roc_auc_score(y_valid, y_pred_valid)
print(f'ROC AUC on delayed sample with new features (start_hour, morning): {auc_valid}')

C_values = np.logspace(-3, 1, 10)
auc_scores = []
for C in C_values:
    lr = LogisticRegression(C=C, random_state=17, solver='liblinear')
    lr.fit(X_train, y_train)
    y_pred_valid = lr.predict_proba(X_valid)[:, 1]
    auc_valid = roc_auc_score(y_valid, y_pred_valid)
    auc_scores.append(auc_valid)

best_idx = np.argmax(auc_scores)
optimal_C = C_values[best_idx]
print(f'Optimal C: {optimal_C}')
print(f'ROC AUC with optimal C: {auc_scores[best_idx]}')