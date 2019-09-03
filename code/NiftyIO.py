### IMPORT PY LIBRARIES
# Library for reading .nii Files
import SimpleITK as sitk
# Python Library 2 manage volumetric data
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt

#######################################################
# .nii Metadata (Origin, Resolution, Orientation)
class Metadata():
    def __init__(self, origen=None, spacing=None, direction=None):
        self.origen = origen
        self.spacing = spacing
        self.direction = direction
        
########################################################
# FUNCTION: readNifty(filePath)
#        
# INPUT: 
# 1> filePath is the full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# OUTPUT: 
# 1> volume_xyz: np.ndarray containing .nii volume
# 2> metadata: .nii metadata         
def readNifty(filePath):
    """
 # INPUT: 
 # 1> filePath is the full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
 # OUTPUT: 
 # 1> volume_xyz: np.ndarray containing .nii volume
 # 2> metadata: .nii metadata 
 #
 # EXAMPLE:
 # 1. Skip metadata output argument
 # import os
 # from PyCode_Session1.NiftyIO import readNifty
 # filePath=os.path.join("Data_Session1","LIDC-IDRI-0001_GT1.nii.gz")
 # vol,_=readNifty(filePath)
    """
    image = sitk.ReadImage(filePath)
    print("Reading Nifty format from {}".format(filePath))
    print("Image size: {}".format(image.GetSize()))

    metadata = Metadata(image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # Converting from SimpleITK image to Numpy array. But also is changed the coordinate systems
    # from the image which use (x,y,z) to the array using (z,y,x).
    volume_zyx = sitk.GetArrayFromImage(image)
    
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))  # to back to the initial xyz coordinate system.


    print("Volume shape: {}".format(volume_xyz.shape))
    print("Minimum value: {}".format(np.min(volume_xyz)))
    print("Maximum value: {}".format(np.max(volume_xyz)))

    #return volume_xyz, metadata     # return two items.
    return np.int32(volume_xyz), metadata
    #return volume_xyz

########################################################
# FUNCTION: saveNifty(volume, metadata, filename)
#        
# INPUT: 
# 1> volume: np.ndarray containing .nii volume
# 2> metadata: .nii metadata (optional).
#    If ommitted default (identity) values are used
# 3> filename is the full path to the output file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# OUTPUT:
#    
def saveNifty(volume, metadata, filename):
    """
    # FUNCTION: saveNifty(volume, metadata, filename)
#        
# INPUT: 
# 1> volume: np.ndarray containing .nii volume
# 2> metadata: .nii metadata (optional).
#    If ommitted default (identity) values are used
# 3> filename is the full path to the output file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
# OUTPUT:
# 
    """
    # Converting from Numpy array to SimpleITK image.

    volume = np.transpose(volume, (2, 1, 0)) # from (x,y,z) to (z,y,x)
    image = sitk.GetImageFromArray(volume)  # It is supposed that GetImageFromArray receive an array with (z,y,x)

    if metadata is not None:
        # Setting some properties to the new image
        image.SetOrigin(metadata.origen)
        image.SetSpacing(metadata.spacing)
        image.SetDirection(metadata.direction)

    sitk.WriteImage(image, filename)
    print("Saving the new image in: {}.".format(filename))


#added jul 18, 2019
    
def construct_histograms(img_3d_NI, img_3d_RD, img_3d_CI, binNum):
    (hist1, edges) = np.histogram(img_3d_NI, bins=np.arange(0,binNum+1), density=False)#NI
    (hist2, edges) = np.histogram(img_3d_RD, bins=np.arange(0,binNum+1), density=False)#RD
    (hist3, edges) = np.histogram(img_3d_CI, bins=np.arange(0,binNum+1), density=False)#CI
    
    joint_2d = np.outer(hist1,hist2).ravel() #Joint: NI/RD    
    concat_2d = np.concatenate([hist1,hist2]) #Concatenation: NI+RD
    concat_3d = np.concatenate([hist1,hist2,hist3]) #Concatenation: NI+RD+CI
    
    return joint_2d, concat_2d, concat_3d

import os
import pandas as pd
def make_train_test_list(src_image_path, src_mask_path, filename):
    # Make two list from the filename, which is an .xlsx file.
    # And return two list with train and test samples each one.
    # test set (Training == 0), train set(Training == 1)
    df = pd.read_excel(filename)
    print("Reading the filename: {}".format(filename))
    train_samples =[] #positive_samples = []
    test_samples = [] #negative_samples = []

    con = df['Training'] == 1 #train set
    df_train = df[con]
    for i in range(0, len(df_train)):
        imageName = df_train.iloc[i][0] + '_GT1_' + str(df_train.iloc[i][1]) + '.nii.gz'
        maskName = imageName.replace('.nii.gz', '_Mask.nii.gz')
        # Check if there is the image and its corresponding mask.
        if (os.path.isfile(os.path.join(src_image_path, imageName)) and
            os.path.isfile(os.path.join(src_mask_path, maskName)) ) == True:
                #Nodule Diagnosis df_train.iloc[i][2]
                train_samples.append((imageName, maskName, df_train.iloc[i][2], df_train.iloc[i][0]))
                print("Added to the sample list: ({}, {})".format(imageName, maskName))
        else:
            print("Discarded pair of samples: ({}, {})".format(imageName, maskName))
    
    #test set
    con = df['Training'] == 0 #test set
    df_test = df[con]
    for i in range(0, len(df_test)):
        imageName = df_test.iloc[i][0] + '_GT1_' + str(df_test.iloc[i][1]) + '.nii.gz'
        maskName = imageName.replace('.nii.gz', '_Mask.nii.gz')
        # Check if there is the image and its corresponding mask.
        if (os.path.isfile(os.path.join(src_image_path, imageName)) and
            os.path.isfile(os.path.join(src_mask_path, maskName)) ) == True:
                #Patient Diagnosis df_test.iloc[i][3]
                #For test set also uses NoduleDiagnosis but to conclude, need to use PatientDiagnosis
                test_samples.append((imageName, maskName, df_test.iloc[i][2], df_test.iloc[i][0]))
                print("Added to the sample list: ({}, {})".format(imageName, maskName))
        else:
            print("Discarded pair of samples: ({}, {})".format(imageName, maskName))  
    print("Length of the sample list: {} rows.".format(len(train_samples) + len(test_samples)))
    return train_samples, test_samples

def make_filename_list(src_image_path, src_mask_path, filename):
    # Make two list from the filename, which is an .xlsx file.
    # And return two list with positive and negative samples each one.
    # Positive = 1 and Negative = 0.
    df = pd.read_excel(filename)
    print("Reading the filename: {}".format(filename))
    positive_samples = []
    negative_samples = []
    for idx in range(0, len(df)):
        imageName = df.iloc[idx][0] + '_GT1_' + str(df.iloc[idx][1]) + '.nii.gz'
        maskName = imageName.replace('.nii.gz', '_Mask.nii.gz')
        # Check if there is the image and its corresponding mask.
        if (os.path.isfile(os.path.join(src_image_path, imageName)) and
            os.path.isfile(os.path.join(src_mask_path, maskName)) ) == True:
                diagnosis = df.iloc[idx][2]#3rd column
                # Positive (maligne) = 1, Negative (benigne) = 0.
                if diagnosis == 1:
                    positive_samples.append((imageName, maskName, 1))
                else:
                    negative_samples.append((imageName, maskName, 0))
                print("Added to the sample list: ({}, {})".format(imageName, maskName))
        else:
            print("Discarded pair of samples: ({}, {})".format(imageName, maskName))
    print("Length of the sample list: {} rows.".format(len(positive_samples) + len(negative_samples)))
    return positive_samples, negative_samples

def saveXLSX(filename, df):
    # write to a .xlsx file.
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    # install xlsxwriter and pip install xlrd.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

from radiomics import featureextractor
from collections import OrderedDict
def getFeatures(imageName, maskName, imageITK, maskITK, y_label, paramPath,patientId):
    # extract features using pyradiomic.
    extractor = featureextractor.RadiomicsFeaturesExtractor(paramPath)
    featureVector = extractor.execute(imageITK, maskITK)
    new_row = {}
    for featureName in featureVector.keys():  # Note that featureVectors is a 'disordered dictionary'
        # print('Computed %s: %s' % (featureName, featureVector[featureName]))
        # print(featureVector[featureName])
        if ('firstorder' in featureName) or ('glszm' in featureName) or \
                ('glcm' in featureName) or ('glrlm' in featureName) or \
                ('gldm' in featureName) or ('shape' in featureName):
            new_row.update({featureName: featureVector[featureName]})
    lst = sorted(new_row.items())  # Ordering the new_row dictionary.
    # Adding some columns
    lst.insert(0, ('diagnosis', y_label))
    lst.insert(0, ('mask_filename', maskName))
    lst.insert(0, ('image_filename', imageName))
    lst.insert(0, ('patientId', patientId))
    od = OrderedDict(lst)
    return (od)

#Aug 16, 2019
def patient_level_evaluation(y_pred, y_actual, scores, threshold):
    TP = 0
    FP = 0 
    TN = 0
    FN = 0
    tpr = []
    fpr = []
    P = sum(y_pred)
    N = len(y_pred) - P
    for t in threshold:
        for i in range(len(scores)):
        #0 - 38
                if (scores[i] > t):
                    #TP: predicted YES, actual YES
                    if y_pred[i]==y_actual[i]==1:
                       TP = TP + 1
                    #FP: predicted YES, actual NO
                    if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
                       FP = FP + 1
                    #TN: predicted NO, actual NO
                    if y_pred[i]==y_actual[i]==0:
                       TN = TN + 1
                    #FN: predicted NO, actual YES
                    if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
                       FN = FN + 1

        fpr.append(FP/float(N))
        tpr.append(TP/float(P)) 
                           
    return (TP, FP, TN, FN, fpr, tpr)

def make_prediction_and_print_scores(model, X_train, y_train, X_test, y_test,c,tile_name):
    model.fit(X_train,y_train)
    #module level predictions
    #y_score=model.fit(X_train, y_train).decision_function(X_test)
    y_pred = model.predict(X_test)
    #patient level predictions
    y_pred_p = list_nodules_to_patients(y_pred)
    y_test_p =  list_nodules_to_patients(y_test)
    
    precision, recall, fscore, support = score(y_test_p, y_pred_p)
    fpr, tpr, threshold = roc_curve(y_test_p, y_pred_p)
    
    roc_auc = auc(fpr, tpr)
    
    print("Applied ",tile_name)
    print("Precision:",precision)
    print("Recall",recall)
    print("AUC",roc_auc)
    print("------ ---- ---- --- ")
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, c, label = tile_name + ', AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Aug 21, 2019
# This functionis to convert 39 nodules to 35 patients
# for the prediction set (based on nodules) and the y_test
def list_nodules_to_patients(predictions):
    y_pred1 = []
    for i in range(len(predictions)):
        if i < 9:
            y_pred1.append(predictions[i])
        if i == 9:
            #predicted P (1)
            if predictions[i] == predictions[i+1] == 1 or predictions[i] != predictions[i+1]:
                y_pred1.append(1)
            #predicted N (0)
            if predictions[i] == predictions[i+1] == 0:
                y_pred1.append(0)
        if i == 10:
            continue
        if 10 < i < 18:
            y_pred1.append(predictions[i])
        if i == 18:    
            #predicted P (1)
            if predictions[i] == predictions[i+1] == 1 or predictions[i] != predictions[i+1]:
                y_pred1.append(1)
            #predicted N (0)
            if predictions[i] == predictions[i+1] == 0:
                y_pred1.append(0)
        if i == 19:
            continue            
        if 19 < i < 32:           
            y_pred1.append(predictions[i])
        if i == 32:            
            #predicted P (1)
            if predictions[i] == predictions[i+1] == 1 or predictions[i] != predictions[i+1]:
                y_pred1.append(1)
            #predicted N (0)
            if predictions[i] == predictions[i+1] == 0:
                y_pred1.append(0)
                
        if i == 33:
            continue
        if 33 < i < 37:  
            y_pred1.append(predictions[i])       
        if i == 37:        
            #predicted P (1)
            if predictions[i] == predictions[i+1] == 1 or predictions[i] != predictions[i+1]:
                y_pred1.append(1)
            #predicted N (0)
            if predictions[i] == predictions[i+1] == 0:
                y_pred1.append(0)
    
    return y_pred1
    '''
    print("Predictions...")
    print('length: ', len(predictions))
    print(predictions)
    
    print("Y_Pred...")
    print("length: ",len(y_pred1))
    print(y_pred1)
    '''
