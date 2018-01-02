import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from timeit import default_timer as timer
# custom module
from modules.data_process import load_data, impact_encode, impact_transform 
from modules.porto_custom_module import gini, normalized_gini, gini_lgb, gini_xgb, save_csv, save_train_log

def porto_ensemble(X, y, test, sub, nrounds=3000, kfold=5, n_models=2,
                   encode_list=None):
    
    blend_train = np.zeros((X.shape[0], n_models)) # number of train data x number of models
    blend_test = np.zeros((test.shape[0], n_models)) # number of test data x number of models 
    normalized_gini_list = np.zeros((n_models * kfold))
    test_copy = test.copy()
    seed = np.random.randint(42*4,42*6, size=n_models)
    # xgb
    print('start XGB training...')
    start = timer()
    params_1 = {
        'eta': 0.07,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree':0.8,
        'objective': 'binary:logistic',
        'scale_pos_weight':1.7,
        'gamma':8,
        'reg_alpha':0.5,
        'reg_lambda':1.3,
        'eval_metric': 'auc',
        'silent': True,
        'seed':seed[0]
    }
                        
    skf1 = KFold(n_splits=kfold, shuffle=True, random_state=322)
    
    for i, (train_index, cv_index) in enumerate(skf1.split(X)):

        print(' xgb kfold: {} of {} : '.format(i+1, kfold))
        X_train, X_eval = X.iloc[train_index,:].copy(), X.iloc[cv_index,:].copy()
        y_train, y_eval = y[train_index], y[cv_index]
        
        
        # impact encoding for the data
        X_train_encoded, X_eval_encoded, test_encoded = impact_transform(encode_list, 
                                                                         X_train, 
                                                                         X_eval, 
                                                                         test_copy, 
                                                                         fold1=20, 
                                                                         fold2=10, 
                                                                         label='target')
        d_train = xgb.DMatrix(X_train_encoded, y_train) 
        d_valid = xgb.DMatrix(X_eval_encoded, y_eval)
         
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        xgb_model = xgb.train(params_1,
                              d_train,
                              nrounds,
                              watchlist,
                              early_stopping_rounds=100,
                              feval=gini_xgb,
                              maximize=True,
                              verbose_eval=50)
        
        blend_train[cv_index,0] = xgb_model.predict(xgb.DMatrix(X_eval_encoded))
        
        normalized_gini_list[i] = normalized_gini(blend_train[cv_index, 0], y_eval)
                       
        blend_test[:,0] += xgb_model.predict(xgb.DMatrix(test_encoded),
                                             ntree_limit=xgb_model.best_ntree_limit+20) / kfold


    end = timer()
    print('XGB training time:', (end - start)/60)

    # lgb
    start = timer()
    print('start LGB training...')
    params_2 = {
        'metric':'auc',
        'learning_rate':0.01,
        'max_depth':4,
        #'max_bin':10,
        'colsample_bytree':0.8,
        'subsample':0.8,
        'objective': 'binary',
        #'feature_fraction': 0.8,
        #'bagging_fraction': 0.9,
        #'bagging_freq': 10,
        #'min_data': 500,
        'seed':seed[1]
    }

    skf2 = KFold(n_splits=kfold, shuffle=True, random_state=42**2)

    for i, (train_index, cv_index) in enumerate(skf2.split(X)):
        
        print(' lgb kfold: {} of {} : '.format(i+1, kfold))
        X_train, X_eval = X.iloc[train_index,:].copy(), X.iloc[cv_index,:].copy()
        y_train, y_eval = y[train_index], y[cv_index]
        
        # impact encoding for the data
        print('Doing impact encoding...')
        X_train_encoded, X_eval_encoded, test_encoded = impact_transform(encode_list, 
                                                                         X_train, 
                                                                         X_eval, 
                                                                         test_copy, 
                                                                         fold1=20, 
                                                                         fold2=10, 
                                                                         label='target')
        lgb_model = lgb.train(params_2,
                              lgb.Dataset(X_train_encoded, label=y_train),
                              nrounds,
                              lgb.Dataset(X_eval_encoded, label=y_eval),
                              verbose_eval=100,
                              feval=gini_lgb,
                              early_stopping_rounds=100)
        
        blend_train[cv_index, 1] = lgb_model.predict(X_eval_encoded)
        normalized_gini_list[i + kfold] = normalized_gini(blend_train[cv_index, 1], y_eval)
        blend_test[:,1] += lgb_model.predict(test_encoded,
                                             num_iteration=lgb_model.best_iteration + 20) / kfold
        
    end = timer()
    print('LGB training time:', (end - start)/60)

    # Ensembling prediction using Logictic regression
    start = timer()
    print('start stacking with LogisticRegression...')
    stack_clf = LogisticRegression()
    stack_clf.fit(blend_train, y)
    sub['target'] = stack_clf.predict_proba(blend_test)[:,1]
    end = timer()
    print('stacking time:', (end - start)/60)
    print('finish training...')
    # saving log
    save_train_log(normalized_gini_list)

    # save predict meta
    save_csv(pd.DataFrame(blend_train), 'blend_train', cols_name=False, fl_format=None)
    save_csv(pd.DataFrame(blend_test), 'blend_test', cols_name=False, fl_format=None)

