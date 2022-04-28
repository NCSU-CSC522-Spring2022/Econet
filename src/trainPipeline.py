from audioop import reverse
from copyreg import pickle
from dataclasses import replace
from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, fbeta_score
from imblearn.under_sampling import TomekLinks 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import pickle


class trainPipeline(): 

    def __init__(self, myPath):
        #read paths of data of all seasons
        self.myFileList = glob.glob(myPath + "*.csv")   

    #apply data resampling technique TOMEKLINKS
    def resampleTrainingData(self, xTrain, yTrain): 
        print("Before resampling")
        print(yTrain.value_counts())
        tl = TomekLinks(n_jobs=-1)
        xTrain1, yTrain1 = tl.fit_resample(xTrain, yTrain)
        print("After resampling")
        print(yTrain1.value_counts())
        return xTrain1, yTrain1

    ##create different machine learning models. We tried XGBoost and RandomForest.
    #Apply grid search for all models for best parameter selection 
    def machineLearningModels(self, modelName): 
        if modelName == "randomForest":
            param_grid = {'n_estimators': [400, 800, 1200],
               'max_features': ['auto'],
               'max_depth': [50, 75, 100, 150, None],
               'min_samples_split': [2,5,10],
               'criterion': ['gini', 'entropy']
            }

            randomForestModel = RandomForestClassifier(random_state=42, n_jobs=16)
            randomForestCVModel = GridSearchCV(estimator=randomForestModel, param_grid=param_grid, cv=3, n_jobs=16, verbose=4, scoring='f1')
            return randomForestCVModel
        
        elif modelName == "xgBoost":
            param_grid = {
                'eta':[0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25],
                'max_depth' : [10, 30, 50, 75]
            }
            XGBoostModel = xgb.XGBClassifier(eval_metric = 'aucpr') #eval metric for xgboost set to aucpr
            xgBoostCVModel = GridSearchCV(estimator=XGBoostModel, param_grid=param_grid, cv=3, n_jobs=16, verbose=4, scoring='f1')
            return xgBoostCVModel

    def trainPipeLine(self):
        for i in range(len(self.myFileList)):   #iterate through list of files (all seasons csvs)
            myFileName = os.path.basename(self.myFileList[i])
            myFileName = os.path.splitext(myFileName)[0]
            readDf = pd.read_csv(self.myFileList[i])
            readDf = readDf.drop(columns=["Ob"])    #drop date time column
            readX = readDf.drop(columns=["target"], axis = 1)
            readY = readDf["target"]

            #splitting the data into stratified train test split test size is set to 20%, s
            XTrain, XVal, yTrain, yVal = train_test_split(readX, readY, stratify = readY, test_size=0.2, random_state=42) 
           
            print("Train shape ", XTrain.shape)
            print("Val shape ", XVal.shape)
                        

            modelCompDict = {}  #empty dictionary to store models
            modelList = ["randomForest"] #list of different models to fit on the data

            #iterate through different models and store them in the dictionary, also print different evaluation metrics.
            #select the model with the best F1 score for all the seasons and save the models
            for i in range(0, len(modelList)): 
                myModelName = modelList[i]
                
                print("Running ", myModelName, " for the file ", myFileName)

                myModel = self.machineLearningModels(myModelName)
                myFit = myModel.fit(XTrain, yTrain)
                myPredict = myModel.predict(XVal)
                myF1 = f1_score(yVal, myPredict)
                myAccuracy = accuracy_score(yVal, myPredict)
                myPrecision = precision_score(yVal, myPredict)
                myRecall = recall_score(yVal, myPredict)
                myf2Score = fbeta_score(yVal, myPredict, average='macro', beta=0.5)
                myConfMatrix = confusion_matrix(yVal, myPredict)
                myClassMat = classification_report(yVal, myPredict)

                modelCompDict[myModel] = [myF1, myAccuracy, myPrecision, myRecall, myf2Score]
                print(myConfMatrix)
                print(myClassMat)
               
            print("**************************************************************************************")
            modelCompDict = dict(sorted(modelCompDict.items(), key=lambda item:item[1][0], reverse=True))
            print("For file ", myFileName, " the best model is ", modelCompDict)

            modelList = list(modelCompDict.keys())
            bestModel = modelList[0]
            modelPath1 = "C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/models/"
            modelPath2 = "/Users/vignesh/Desktop/Projects/Econet/models/"
            pickle.dump(bestModel, file=open(modelPath2 + myFileName + ".sav",'wb')) #save the best model
            

if __name__ == "__main__":
    path2 = "/Users/vignesh/Desktop/Projects/Econet/trainData/"
    myTrainObj = trainPipeline(path2)
    myTrainObj.trainPipeLine()