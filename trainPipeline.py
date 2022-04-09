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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import pickle

class trainPipeline():

    def __init__(self, myPath):
        self.myFileList = glob.glob(myPath + "*.csv")

    
    def machineLearningModels(self, modelName):
        if modelName == "randomForest":
            randomForestModel = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=16)
            return randomForestModel
        
        elif modelName == "xgBoost":
            XGBoostModel = xgb.XGBClassifier()
            return XGBoostModel
    
        elif modelName == "knnClassifier":
            knnModel = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', n_jobs=16)
            return knnModel
    
    def trainPipeLine(self):
        
        for i in range(len(self.myFileList)):
            myFileName = os.path.basename(self.myFileList[i])
            myFileName = os.path.splitext(myFileName)[0]
            readDf = pd.read_csv(self.myFileList[i], index_col=["Ob", "Station"])
            readDf = readDf.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
            readX = readDf.drop(columns=["target"], axis = 1)
            readY = readDf["target"]
            XTrain, XVal, yTrain, yVal = train_test_split(readX, readY, stratify = readY, test_size=0.3, random_state=42)


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

                modelCompDict[myModel] = [myF1, myAccuracy, myPrecision, myRecall]
                print("**************************************************************************************")

            
            modelCompDict = dict(sorted(modelCompDict.items(), key=lambda item:item[1][0], reverse=True))
            print("For file ", myFileName, " the best model is ", modelCompDict)
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
    myTrainObj = trainPipeline("/home/atharva/Atharva/NCSU/Sem2/ALDA/project/EcoNet1/Econet/data/splitData/seasonSplitData/")
    myTrainObj.trainPipeLine()