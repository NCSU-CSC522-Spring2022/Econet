from audioop import reverse
from copyreg import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, fbeta_score
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import pickle
from imblearn.combine import SMOTEENN, SMOTETomek

class trainPipeline():

    def __init__(self, myPath):
        self.myFileList = glob.glob(myPath + "*.csv")

    def resampleTrainingData(self, xTrain, yTrain):
        print("Before resampling")
        print(yTrain.value_counts())
        tl = TomekLinks(n_jobs=16)
        xTrain, yTrain = tl.fit_resample(xTrain, yTrain)
        # t2 = SMOTETomek(n_jobs=-1)
        # xTrain, yTrain = t2.fit_resample(xTrain, yTrain)
        print("After resampling")
        print(yTrain.value_counts())
        return xTrain, yTrain


    def machineLearningModels(self, modelName):
        if modelName == "randomForest":
            param_grid = {'n_estimators': [400, 800, 1200],
               'max_features': ['auto'],
               'max_depth': [50, 75, 100, 150, None],
               'min_samples_split': [2,5,10],
            #    'min_samples_leaf': [1,2,4],
            #    'bootstrap': [True, False],
               'criterion': ['gini', 'entropy']
            }

            randomForestModel = RandomForestClassifier(random_state=42, n_jobs=16)
            randomForestCVModel = GridSearchCV(estimator=randomForestModel, param_grid=param_grid, cv=3, n_jobs=16, verbose=4, scoring=f1_score)
            return randomForestCVModel
        
        elif modelName == "xgBoost":
            param_grid = {
                'eta':[0.01, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25],
                'max_depth' : [10, 30, 50, 75]
            }
            XGBoostModel = xgb.XGBClassifier()
            return XGBoostModel
    
        elif modelName == "knnClassifier":
            knnModel = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', n_jobs=16)
            return knnModel
    
    def trainPipeLine(self):
        # print(self.myFileList)
        for i in range(len(self.myFileList)):
            myFileName = os.path.basename(self.myFileList[i])
            myFileName = os.path.splitext(myFileName)[0]
            readDf = pd.read_csv(self.myFileList[i]) #, index_col=["Station"]
            readDf = readDf.drop(columns=["Station", "Ob"])
            print(readDf.shape)
            readX = readDf.drop(columns=["target"], axis = 1)
            readY = readDf["target"]
            XTrain, XVal, yTrain, yVal = train_test_split(readX, readY, stratify = readY, test_size=0.3, random_state=42)


            XTrain, yTrain = self.resampleTrainingData(XTrain, yTrain)
            modelCompDict = {}
            modelList = ["randomForest", "xgBoost"]
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
            pickle.dump(bestModel, file=open("/Users/vignesh/Desktop/Projects/Econet/models/" + myFileName + ".sav",'wb'))
            
            # myAccuracy1 = accuracy_score(yVal, myPredict1)
            # myF11 = f1_score(yVal, myPredict1)

            # if myAccuracy1 >= myAccuracy2 and myF11 >= myF12:
            #     pickle.dump(myModel1, file=open(myFileName + ".sav",'wb'))
            
            # elif myAccuracy1 <= myAccuracy2 and myF11 <= myF12:
            #     pickle.dump(myModel2, file = open(myFileName + ".sav",'wb'))
            
            # elif myF11 >= myF12:
            #     pickle.dump(myModel1, file = open(myFileName + ".sav",'wb'))
            
            # elif myF11 <= myF12:
            #     pickle.dump(myModel2, file = open(myFileName + ".sav",'wb'))

if __name__ == "__main__":
    myTrainObj = trainPipeline("/Users/vignesh/Desktop/Projects/Econet/splitDataMod/")
    myTrainObj.trainPipeLine()