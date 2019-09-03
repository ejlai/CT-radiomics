#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:31:43 2019
Updated: Aug 15, 2019

change log:
    Test MinMaxScaler(), RobustScaler() instead of StandardScaler()
    8/22: nodule level to patient level evaluations
"""

#Logistic Regression tuning
import pickle
#from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn


#pickle files
in_dir = '../db/pickles/'


def perform_evaluation(X, y, test, c, k, mytext):
# pass initial state to generate same indexes
    #model = RF(random_state=1, max_depth=m, n_estimators=n)
    model = SVC(random_state=1, C = c, kernel=k)
    model.probability = True
    model.fit(X, y)    
    predictions=model.predict(test)
    precision, recall, fscore, support = score(y_test, predictions)
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(test)[:,1])#model.predict_proba(X_test)[:,1]
    auc1 = auc(fpr, tpr)
    print()
    print(mytext)
    print("Predictions: {}".format(predictions))
    print("Y_test: {}".format(y_test))
    print()
    print('precision: {}'.format(precision)) #class 0, 1
    print('recall: {}'.format(recall)) #class 0, 1
    #print('fscore: {}'.format(fscore))
    #print('support: {}'.format(support))
    print('roc_auc_score: {:.2f}'.format(auc1))
    print()

#b='_default' #bins = [12, 20, 28, 36, 44,'_default']
#joint2d, concat2d, concat3d, y_train
b = 44
#train set
train_picklefile= in_dir + 'train_p42g_r1_v3_bins'+str(b)+'.pickle'
#load Train/Validate/Test dataset
with open(train_picklefile,'rb') as f:
    #These are list and elemnet of the list is an array.
    X_train_joint, X_train_cc2d, X_train_cc3d, y_train = pickle.load(f)

#test set
test_picklefile= in_dir + 'test_p42g_r1_v3_bins'+str(b)+'.pickle'
#load Train/Validate/Test dataset
with open(test_picklefile,'rb') as f:
    #These are list and elemnet of the list is an array.
   X_test_joint, X_test_cc2d, X_test_cc3d, y_test = pickle.load(f)

'''Standardize the Data'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

#MinMaxScaler() RobustScaler() StandardScaler()

#joint
scaler = MinMaxScaler()
# Fit on training set only.
scaler.fit(X_train_joint)
# Apply transform to both the training set and the test set.
X_train_joint = scaler.transform(X_train_joint)
X_test_joint = scaler.transform(X_test_joint)    

#concat2d
scaler = MinMaxScaler()
scaler.fit(X_train_cc2d)
# Apply transform to both the training set and the test set.
X_train_cc2d = scaler.transform(X_train_cc2d)
X_test_cc2d = scaler.transform(X_test_cc2d)        

#concat3d
scaler = MinMaxScaler()
scaler.fit(X_train_cc3d)
# Apply transform to both the training set and the test set.
X_train_cc3d = scaler.transform(X_train_cc3d)
X_test_cc3d = scaler.transform(X_test_cc3d)   


'''Feature reduction'''
#joint set
pca = PCA(.95) #PCA (.75) #3 components, 88 features, 52 samples
pca.fit(X_train_joint)
#Apply the mapping (transform) to both set.
X_train_joint_pca = pca.transform(X_train_joint)
X_test_joint_pca = pca.transform(X_test_joint)
   #cc2d set
pca = PCA(.95) #PCA (.75) #3 components, 88 features, 52 samples
pca.fit(X_train_cc2d)
#Apply the mapping (transform) to both set.
X_train_cc2d_pca = pca.transform(X_train_cc2d)
X_test_cc2d_pca = pca.transform(X_test_cc2d)

   #cc3d set
pca = PCA(.95) #PCA (.75) #3 components, 88 features, 52 samples
pca.fit(X_train_cc3d)
#Apply the mapping (transform) to both set.
X_train_cc3d_pca = pca.transform(X_train_cc3d)
X_test_cc3d_pca = pca.transform(X_test_cc3d)

'''perform LDA of 2 dimensions on the data'''

#joint set
lda = LDA(n_components=2)
lda_object = lda.fit(X_train_joint, y_train) # creates an LDA object for (inputs, targets)
X_train_joint_lda = lda_object.transform(X_train_joint)
X_test_joint_lda = lda_object.transform(X_test_joint)

#cc2d set
lda = LDA(n_components=2)
lda_object1 = lda.fit(X_train_cc2d, y_train) # creates an LDA object for (inputs, targets)
X_train_cc2d_lda = lda_object1.transform(X_train_cc2d)
X_test_cc2d_lda = lda_object1.transform(X_test_cc2d)

#cc3d set
lda = LDA(n_components=2)
lda_object2 = lda.fit(X_train_cc3d, y_train) # creates an LDA object for (inputs, targets)
X_train_cc3d_lda = lda_object2.transform(X_train_cc3d)
X_test_cc3d_lda = lda_object2.transform(X_test_cc3d)

del lda_object,lda_object1,lda_object2

'''Also perform PCA before doing LDA'''

#joint set
lda_object = lda.fit(X_train_joint_pca, y_train)
X_train_joint_pca_lda = lda_object.transform(X_train_joint_pca)
X_test_joint_pca_lda = lda_object.transform(X_test_joint_pca)

#cc2d set
lda_object = lda.fit(X_train_cc2d_pca, y_train)
X_train_cc2d_pca_lda = lda_object.transform(X_train_cc2d_pca)
X_test_cc2d_pca_lda = lda_object.transform(X_test_cc2d_pca)

#cc3d set
lda_object = lda.fit(X_train_cc3d_pca, y_train)
X_train_cc3d_pca_lda = lda_object.transform(X_train_cc3d_pca)
X_test_cc3d_pca_lda = lda_object.transform(X_test_cc3d_pca)    

del lda_object

'''Make prediction with tuned model'''
'''
 - Concat2d NI+RD: BEST PARAMS: {'C': 1, 'kernel': 'rbf'}
 - Concat3d NI+RD+CI: BEST PARAMS: {'C': 0.1, 'kernel': 'rbf'}
 - Joint2d NI/RD: BEST PARAMS: {'C': 0.1, 'kernel': 'rbf'}
 
'''
c=1
k='rbf'
model = SVC(random_state=1, C = c, kernel=k)

print("Joint set ")
c=1
k='rbf'
model = SVC(random_state=1, C = c, kernel=k)
#Joint2d NI/RD: BEST PARAMS: {'C': 0.1, 'kernel': 'rbf'}
from NiftyIO import make_prediction_and_print_scores #params: (model, X_train, y_train, X_test, y_test,c,tile_name)

make_prediction_and_print_scores(model, X_train_joint, y_train, X_test_joint, y_test,'c','None')
make_prediction_and_print_scores(model, X_train_joint_pca, y_train, X_test_joint_pca, y_test,'y','PCA')
make_prediction_and_print_scores(model, X_train_joint_lda, y_train, X_test_joint_lda, y_test,'m','LDA')
make_prediction_and_print_scores(model, X_train_joint_pca_lda, y_train, X_test_joint_pca_lda, y_test,'k','PCA+LDA')

print('-------------------------')

print("Concat2D set ")
c=0.1
model = SVC(random_state=1, C = c, kernel=k)

#Concat2d NI+RD: BEST PARAMS: {'C': 1, 'kernel': 'rbf'}
make_prediction_and_print_scores(model, X_train_cc2d, y_train, X_test_cc2d, y_test,'c','None')
make_prediction_and_print_scores(model, X_train_cc2d_pca, y_train, X_test_cc2d_pca, y_test,'y','PCA')
make_prediction_and_print_scores(model, X_train_cc2d_lda, y_train, X_test_cc2d_lda, y_test,'m','LDA')
make_prediction_and_print_scores(model, X_train_cc2d_pca_lda, y_train, X_test_cc2d_pca_lda, y_test,'k','PCA+LDA')

print('-------------------------')    
#Concat3d NI+RD+CI: BEST PARAMS: {'C': 0.1, 'kernel': 'rbf'}
print("Concat3D set ")
make_prediction_and_print_scores(model, X_train_cc3d, y_train, X_test_cc3d, y_test,'c','None')
make_prediction_and_print_scores(model, X_train_cc3d_pca, y_train, X_test_cc3d_pca, y_test,'y','PCA')
make_prediction_and_print_scores(model, X_train_cc3d_lda, y_train, X_test_cc3d_lda, y_test,'m','LDA')
make_prediction_and_print_scores(model, X_train_cc3d_pca_lda, y_train, X_test_cc3d_pca_lda, y_test,'k','PCA+LDA')

