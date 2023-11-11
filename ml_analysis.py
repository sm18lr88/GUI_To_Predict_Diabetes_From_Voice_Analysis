# Created 2023 by Jaycee Kaufman, Klick Inc. 
# Contact: jmorgankaufman@klick.com

import pandas as pd 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold

'''
Versions for Packages
pandas==1.5.2
numpy==1.23.5
scikit-learn==1.2.0
'''

def pipeline_fun(df,feats,h_train_names,d_train_names,h_test_names,d_test_names,i,mod_type):
    '''
    df: dataframe containing voice data, must contain all features in feats as column names, and "ID" as column name for participant ID
    feats: list of voice features to be used in training
    h_train_names: list containing healthy participant ids to use in training
    h_test_names: list containing healthy participant ids to use in testing
    d_train_names: list containing t2dm participant ids to use in training
    d_test_names: list containing t2dm participant ids to use in testing
    i: iteration, natural number (>0)
    mod_type: model to use, must be one of {"nb","lr","svm"}, corresponding to Naive Bayes, Logistic Regression, and Support Vector Machine, respectively

    output: test (a dataframe containing the prediction results for the test data)
    '''
    
    # Prepping Data, Splitting Datasets

    h_train = df[df['ID'].isin(h_train_names)]
    h_test = df[df['ID'].isin(h_test_names)]
    d_train = df[df['ID'].isin(d_train_names)]
    d_test = df[df['ID'].isin(d_test_names)]

    train_df = pd.concat([h_train,d_train])
    test_df = pd.concat([h_test,d_test])

    feat_df = train_df[feats]
    labels = train_df['Diagnosis']

    test_feat = test_df[feats]
    test=test_df[["ID","Diagnosis"]].copy()

    # ML

    if mod_type=="nb":
        clf = GaussianNB()
    elif mod_type=="lr":
        clf = LogisticRegression()
    elif mod_type=="svm":
        clf = SVC(probability=True)

    pipe = make_pipeline(StandardScaler(),clf)
    pipe.fit(feat_df, labels)
    test_pred = pipe.predict_proba(test_feat)
    test[f"Fold {i} Probability"] = list(test_pred[:,1])

    return test

def cross_val(num_folds,df,feats,mod):
    '''
    function to perform cross validation (CV) on voice data

    Inputs:
    num_folds - number of folds for CV
    df - voice feature dataframe. must contain columns "ID" (participant ID), "Diagnosis" (0 for non-diabetic, 1 for t2dm), and voice feature columns
    feats - list of features to use in ML. must be columns in df
    mod - model type. must be one of {"nb","lr","svm"}, corresponding to Naive Bayes, Logistic Regression, and Support Vector Machine, respectively

    Output:
    final_df - dataframe containing prediction probability test results for each fold
    '''

    final_df = df[["ID","Diagnosis"]].copy()

    nd = np.array(df[df["Diagnosis"]==0]["ID"].unique()) # non-diabetic
    t2d = np.array(df[df["Diagnosis"]==1]["ID"].unique()) # t2dm

    # Splitting non-diabetic and t2dm data into folds
    kf = KFold(n_splits=num_folds,shuffle=True)
    all_nd_split = kf.split(nd)
    all_d_split = kf.split(t2d)

    for i,val in enumerate(zip(all_nd_split,all_d_split)):
        (nd_train_index, nd_test_index),(d_train_index, d_test_index) = val

        nd_train_names = nd[nd_train_index]
        nd_test_names = nd[nd_test_index]

        d_train_names = t2d[d_train_index]
        d_test_names = t2d[d_test_index]
        
        pred_df = pipeline_fun(df, feats,nd_train_names,d_train_names,nd_test_names,d_test_names,i,mod)
        final_df = pd.concat([final_df,pred_df[f"Fold {i} Probability"]],axis=1)

    return final_df