'''
This one use impact encoding

List of features for the data set: 
Index(['id', 'target', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',
'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',
'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12',
'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], dtype='object')
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from timeit import default_timer as timer

def impact_encode(cat_feat=None, data=None, fold1=20, fold2=10, label='target'):
    '''Impact encode based on likelihood, works only with data contains no NaN
    field.  dict_encoded is a dictionary that maps each categorical value (key)
    with its associated likelihood.
    
    Based on a Kaggle top user's comment: Let's say we have 20-fold cross
    validation. we need somehow to calculate mean value of the feature for #1
    fold using information from #2-#20 folds only. So, you take #2-#20 folds,
    create another cross validation set within it (I did 10-fold).  calculate
    means for every leave-one-out fold (in the end you get 10 means). You
    average these 10 means and apply that vector for your primary #1 validation
    set. Repeat that for remaining 19 folds.
    
    The triple loops is very slow, but currently no solution yet to be found.
    
    Another warning: apply this function after loading the data, inside the
    cross validation process. Otherwise you will have information leakage.

    '''
    seed_kf = np.random.randint(42, 42*3, size=2)
    col = [cat_feat, label]
    ref_data = data[col]
    cat_value = sorted(ref_data.iloc[:,0].unique())
    vcount = len(cat_value)
    kf1 = KFold(n_splits=fold1, shuffle=True, random_state=seed_kf[0])
    encoded = np.zeros(vcount)
    
    for ref1, oof1 in kf1.split(ref_data):
        oof1_default_mean = ref_data.iloc[ref1][label].mean()
        temp_value = np.zeros(vcount)
        kf2 = KFold(n_splits=fold2, shuffle=True, random_state=seed_kf[1])
        
        for ref2, oof2 in kf2.split(ref_data.iloc[ref1][label]):
            oof2_mean = ref_data.iloc[ref2].groupby(cat_feat)[label].mean()
            for cat in cat_value:
                if cat in oof2_mean.index:
                    temp_value[cat_value.index(cat)] += oof2_mean.loc[cat] / fold2
                else:
                    temp_value[cat_value.index(cat)] += oof1_default_mean / fold2
        encoded += temp_value / fold1
    
    return dict(zip(cat_value, encoded))

def impact_transform(cat_list=None, train_data=None, valid_data=None,
                     test_data=None, fold1=20, fold2=10, label='target'):
    print('Doing impact encoding...')
    for i, cat in enumerate(cat_list):
        dict_encoded = impact_encode(cat, train_data, fold1, fold2, label)
        key_value = list(dict_encoded.keys())
        for key in key_value:
            train_data.loc[train_data[cat] == key, cat + '_imp'] = dict_encoded[key]
            valid_data.loc[valid_data[cat] == key, cat + '_imp'] = dict_encoded[key]
            test_data.loc[test_data[cat] == key, cat + '_imp'] = dict_encoded[key]

    cat_new = cat_list + [label] # used to drop column in train and valid data set
    return train_data.drop(cat_new, axis=1).values, valid_data.drop(cat_new, axis=1).values, test_data.drop(cat_list, axis=1).values

def load_data():
    data_path = '../data/'
    start = timer()
    print('load data...')
    porto_train = pd.read_csv(data_path + 'train.csv', na_values=-1)
    porto_test = pd.read_csv(data_path + 'test.csv', na_values=-1)
    # calc_col = [c for c in porto_train.columns if 'ps_calc_' in c]
    # drop_col is made based on the feature important images
    
    features = [ # based on feature importance when running RandomForestClassifier
        'ps_car_13', 'ps_reg_03', 'ps_ind_05_cat', 'ps_ind_03', 'ps_ind_15',
        'ps_reg_02', 'ps_car_14', 'ps_car_12', 'ps_car_01_cat', 'ps_car_07_cat',
        'ps_ind_17_bin', 'ps_car_03_cat', 'ps_reg_01', 'ps_car_15', 'ps_ind_01',
        'ps_ind_16_bin', 'ps_ind_07_bin', 'ps_car_06_cat', 'ps_car_04_cat',
        'ps_ind_06_bin', 'ps_car_09_cat', 'ps_car_02_cat', 'ps_ind_02_cat',
        'ps_car_11', 'ps_car_05_cat', 'ps_calc_09', 'ps_calc_05',
        'ps_ind_08_bin', 'ps_car_08_cat', 'ps_ind_09_bin', 'ps_ind_04_cat',
        'ps_ind_18_bin', 'ps_ind_12_bin', 'ps_ind_14',
    ]
    
    # Creating new features
    for df in [porto_train, porto_test]:
        
        # df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['NaN_count'] = df.isnull().sum(axis=1)
        df.fillna(-1, inplace=True)
        
    cat_feat = [c for c in features if '_cat' in c]
    cat_feat_encoded = [cat for cat in cat_feat if len(list(porto_train[cat].unique())) > 2]
    drop_col = [c for c in porto_train.columns if c not in (features + ['target'])]

    '''
    # One hot encoded categorical features
    ohe = OneHotEncoder()
    train_cat_ohe = porto_train[cat_feat_encoded].replace(-1, 100000).values
    test_cat_ohe = porto_test[cat_feat_encoded].replace(-1, 100000).values
    
    ohe.fit(train_cat_ohe)
    train_cat_ohe = ohe.transform(train_cat_ohe).toarray()
    test_cat_ohe = ohe.transform(test_cat_ohe).toarray()
    
    # Convert train & test data to numpy array
    X = porto_train.drop((['target'] + drop_col), axis=1).values
    test = porto_test.drop(drop_col, axis=1).values
    
    X = np.concatenate((X, train_impact_encoded), axis=1)
    test = np.concatenate((test, test_impact_encoded), axis=1)
    '''

    # Convert train & test data to numpy array
    X = porto_train.drop(drop_col, axis=1)
    test = porto_test.drop(drop_col, axis=1)
    y = porto_train['target'].values
    
    # Create empty submission file with only test ID columns
    sub = porto_test['id'].to_frame()
    sub['target'] = 0
    print('finish loading data!')
    end = timer()
    print('Data loading time:', (end - start)/60)
    return cat_feat_encoded, X, y, test, sub # y is a numpy array instead of pandas series
