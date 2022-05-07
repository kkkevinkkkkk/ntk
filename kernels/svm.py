import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
import pandas as pd

def svc_search(ntk, labels, train_fold_idx, test_fold_idx):
    gram = ntk
    # C_list = np.logspace(-2, 4, 120)
    C_list = np.logspace(-2, 4, 10)
    svc = SVC(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svc, {'C': C_list},
                       cv=zip(train_fold_idx, test_fold_idx),
                       n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram, labels)
    df = pd.DataFrame({'C': C_list,
                       'train': clf.cv_results_['mean_train_score'],
                       'test': clf.cv_results_['mean_test_score']},
                      columns=['C', 'train', 'test'])

    # also normalized gram matrix
    gram_nor = np.copy(gram)
    gram_diag = np.sqrt(np.diag(gram_nor))
    gram_nor /= gram_diag[:, None]
    gram_nor /= gram_diag[None, :]

    svc = SVC(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svc, {'C': C_list},
                       cv=zip(train_fold_idx, test_fold_idx),
                       n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram_nor, labels)
    df_nor = pd.DataFrame({'C': C_list,
                           'train': clf.cv_results_['mean_train_score'],
                           'test': clf.cv_results_['mean_test_score']},
                          columns=['C', 'train', 'test'])

    df['normalized'] = False
    df_nor['normalized'] = True
    all_df = pd.concat([df, df_nor])[['C', 'normalized', 'train', 'test']]

    return all_df

def svr_search(gram, labels, train_fold_idx, test_fold_idx):
    # C_list = np.logspace(-2, 4, 120)
    C_list = np.logspace(-4, 4, 10)

    svr = SVR(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svr, {'C': C_list},
                       cv=zip(train_fold_idx, test_fold_idx),
                       scoring = 'neg_mean_squared_error',
                       n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram, labels)
    df = pd.DataFrame({'C': C_list,
                       'train': clf.cv_results_['mean_train_score'],
                       'test': clf.cv_results_['mean_test_score']},
                      columns=['C', 'train', 'test'])

    # also normalized gram matrix
    gram_nor = np.copy(gram)
    gram_diag = np.sqrt(np.diag(gram_nor))
    gram_nor /= gram_diag[:, None]
    gram_nor /= gram_diag[None, :]

    svr = SVR(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svr, {'C': C_list},
                       cv=zip(train_fold_idx, test_fold_idx),
                       scoring = 'neg_mean_squared_error',
                       n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram_nor, labels)
    df_nor = pd.DataFrame({'C': C_list,
                           'train': clf.cv_results_['mean_train_score'],
                           'test': clf.cv_results_['mean_test_score']},
                          columns=['C', 'train', 'test'])

    df['normalized'] = False
    df_nor['normalized'] = True
    all_df = pd.concat([df, df_nor])[['C', 'normalized', 'train', 'test']]
    return all_df



