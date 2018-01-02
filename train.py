import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# load custom module (inside this same folder)
from modules.data_process import load_data, impact_encode, impact_transform 
from modules.porto_custom_module import gini, normalized_gini, gini_lgb, gini_xgb, save_csv, save_train_log
from modules.porto_clf import porto_ensemble

if __name__ == "__main__":

    # load data
    cat_feat_encoded, X, y, test, sub = load_data()

    # seed is the random seed for Kfold CV
    porto_ensemble(X, y, test, sub, nrounds=3000, kfold=5, encode_list=cat_feat_encoded)
    
    # saving prediction
    save_csv(sub, 'submission-ensemble')
    
    # finish
    print('done!')
