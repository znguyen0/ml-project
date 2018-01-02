import pandas as pd
import numpy as np
import datetime

def gini(pred, y):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def normalized_gini(pred, y):
    return gini(pred, y) / gini(y, y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', normalized_gini(pred, y)

def gini_lgb(pred, dtrain):
    y = list(dtrain.get_label())
    score = normalized_gini(pred, y)
    return 'gini', score, True

def save_csv(dataframe, file_begin, cols_name=True, fl_format='%.8f'):
    '''
    For saving and storing prediction file systematically
    '''
    print('save', file_begin ,'file...')
    dataframe.to_csv(path_or_buf= file_begin +
                     datetime.datetime.now().strftime("-%y%m%d-%H%M%S") + '.csv',
                     index=False,
                     float_format=fl_format,
                     header=cols_name)
    print('finish saving!')

def save_train_log(log_list):
    '''For saving log of mean and std of training rounds
    '''
    print('saving log...')
    with open('porto_seguro.log', 'a') as f:
        print(datetime.datetime.now().strftime("%y%m%d-%H%M%S") +
              ' | CV_gini has mean={:.6f} '
              'and std={:.6f} | {}'.format(np.mean(log_list),
                                         np.std(log_list),
                                           log_list.tolist()), file=f)
    print('done saving log!')
