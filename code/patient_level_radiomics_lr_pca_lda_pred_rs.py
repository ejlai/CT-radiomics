#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:31:43 2019
change log:
    8/15: Test MinMaxScaler(), RobustScaler() instead of StandardScaler()
    8/21: from nodule level to patient level evaluations

"""
#depress warnings
def warn(*args, **kwargs): pass

import warnings
warnings.warn = warn

#file that creates the pickle: prepare_radiomics_features_26_train_test_pickles.py
picklefile="../db/selected_radiomics_train_test_sets.pickle"

#load dataset
import pickle
with open(picklefile,'rb') as f:
    #These are list and elemnet of the list is an array.
    #X_train26, X_test26, y_train26, y_test26
    X_train, X_test, y_train, y_test = pickle.load(f)

'''standardize data'''
from sklearn.preprocessing import RobustScaler #StandardScaler, MinMaxScaler
scaler = RobustScaler()#MinMaxScaler()  StandardScaler()

# fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''feature reduction'''
"Dimensionality reduciton #1 - PCA"
from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X_train)

# Apply the transform to both set
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

"Dimensionality reduciton #2 - LDA"
# Perform LDA of 2 dimensions on the data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
lda_object = lda.fit(X_train, y_train) # creates an LDA object for (inputs, targets)

"Dimensionality reduciton #3 - PCA + LDA"
# Returns a new basis*data matrix, like PCA does
X_train_lda = lda_object.transform(X_train)
X_test_lda = lda_object.transform(X_test)
del lda_object

# Also perform PCA before doing LDA,
lda_object = lda.fit(X_train_pca, y_train)
X_train_pca_lda = lda_object.transform(X_train_pca)
X_test_pca_lda = lda_object.transform(X_test_pca)
del lda_object

'''Make prediction with tuned model'''
from sklearn.linear_model import LogisticRegression
"""
LR tuning with default bins number dataset found BEST PARAMS: {'C': 0.001}
"""

model = LogisticRegression(random_state=1, C=0.001)

from NiftyIO import make_prediction_and_print_scores #params: (model, X_train, y_train, X_test, y_test,c,tile_name)

"Make prediction and print score, graph"
make_prediction_and_print_scores(model, X_train, y_train, X_test, y_test,'c','None')
make_prediction_and_print_scores(model, X_train_pca, y_train, X_test_pca, y_test,'y','PCA')
make_prediction_and_print_scores(model, X_train_lda, y_train, X_test_lda, y_test,'m','LDA')
make_prediction_and_print_scores(model, X_train_pca_lda, y_train, X_test_pca_lda, y_test,'k','PCA+LDA')
