from cgi import test
from copyreg import pickle
from multiprocessing import dummy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import pickle
import datetime


class TestPipeLine():
    def __init__(self, modelsBasePath, testFilePath):
        self.basePath = modelsBasePath
        self.modelFileList = glob.glob(self.basePath + "*.sav")
        self.testFilePath = testFilePath
        
        self.fall1Model = None
        self.fall2Model = None
        self.summer1Model = None
        self.summer2Model = None
        self.springModel = None
        self.winterModel = None
    
    def loadModels(self):
        # print(self.modelFileList)
        for modelPath in self.modelFileList:
            print(modelPath)
            springModelLoc = open(modelPath, 'rb')
            model = pickle.load(springModelLoc)
            
            if "spring" in modelPath:
                self.springModel = model
            elif "summer1" in modelPath:
                self.summer1Model = model
            elif "summer2" in modelPath:
                self.summer2Model = model
            elif "fall1" in modelPath:
                self.fall1Model = model
            elif "fall2" in modelPath:
                self.fall2Model = model
            elif "winter" in modelPath:
                self.winterModel = model
        
    # Function not used    
    def myDummyOHE(self):
        myTestDfX = pd.read_csv(self.testFilePath)
        myTestDfX["Ob"] = pd.to_datetime(myTestDfX["Ob"], infer_datetime_format=True).dt.time
        dummyDf = pd.get_dummies(myTestDfX.measure, prefix='measure')
        print("Shape of dummyDf: ", dummyDf.shape)
        print("dummyDf columns: ", dummyDf.columns)
        encodedData = pd.concat([myTestDfX[["Station", "Ob", "value", "R_flag", "I_flag", "Z_flag", 'B_flag']], dummyDf], axis = 1)
        # encodedData = encodedData.set_index("Ob")
        return encodedData

    # Function not used
    def splitTestDf(self, myTestDfX):

        springStart = datetime.datetime(2021, 3, 1)
        springEnd = datetime.datetime(2021, 5, 31)
        winterStart = datetime.datetime(2021, 12, 1)
        winterEnd = datetime.datetime(2021, 12, 31)
        winter2Start = datetime.datetime(2021, 1, 1)
        winter2End = datetime.datetime(2021, 2, 28)


        springDf = myTestDfX[(myTestDfX["Ob"] >= springStart) & (myTestDfX["Ob"] <= springEnd)]

        summerDf = myTestDfX[(myTestDfX["Ob"] >= summerStart) & (myTestDfX["Ob"] <= summerEnd)]
        fallDf = myTestDfX[(myTestDfX["Ob"] >= fallStart) & (myTestDfX["Ob"] <= fallEnd)]
        winterDf = myTestDfX[(myTestDfX["Ob"] >= winterStart) & (myTestDfX["Ob"] <= winterEnd)]
        winter2Df = myTestDfX[(myTestDfX["Ob"] >= winter2Start) & (myTestDfX["Ob"] <= winter2End)]
        winterDf = pd.concat([winterDf, winter2Df])

        return springDf, summerDf, fallDf, winterDf
    

    # Function not used
    def predictClasses(self, dataFrame):
        springDf, summer1Df, summer2Df, fall1Df, fall2Df, winterDf = self.splitTestDf(dataFrame)

        # Drop OB and station
        springDf = springDf.drop(["Ob", "Station"], axis = 1)
        springPredict = self.springModel.predict_proba(springDf)
        springPredictDf = pd.DataFrame(springPredict[:,1], index=springDf.index)

        summer1Df = summer1Df.drop(["Ob", "Station"], axis = 1)
        summer1Predict = self.summer1Model.predict_proba(summer1Df)
        summer1PredictDf = pd.DataFrame(summer1Predict[:,1], index=summer1Df.index)

        summer2Df = summer2Df.drop(["Ob", "Station"], axis = 1)
        summer2Predict = self.summer2Model.predict_proba(summer2Df)
        summer2PredictDf = pd.DataFrame(summer2Predict[:,1], index=summer2Df.index)

        fall1Df = fall1Df.drop(["Ob", "Station"], axis = 1)
        fall1Predict = self.fall1Model.predict_proba(fall1Df)
        fall1PredictDf = pd.DataFrame(fall1Predict[:,1], index=fall1Df.index)

        fall2Df = fall2Df.drop(["Ob", "Station"], axis = 1)
        fall2Predict = self.fall2Model.predict_proba(fall2Df)
        fall2PredictDf = pd.DataFrame(fall2Predict[:,1], index=fall2Df.index)

        winterPredict = self.winterModel.predict_proba(winterDf)
        winterPredictDf = pd.DataFrame(winterPredict[:,1], index=winterDf.index)

        concatenatedDf = springPredictDf.concat([summer1PredictDf, summer2PredictDf, fall1PredictDf, fall2PredictDf, winterPredictDf], axis = 0)

        # concatenatedDf.to_csv("predictedValues.csv")
        # print(springDf.head())
        # print(summerDf.head())
        # print(fallDf.head())
        # print(winterDf.head())
        return concatenatedDf


    def getSeasonValue(self, eachDataPoint):
        obValue = eachDataPoint["Ob"].values[0]
        # print(obValue)
        obValue = pd.to_datetime(obValue, infer_datetime_format=True)

        springStart = datetime.datetime(2021, 3, 1)
        springEnd = datetime.datetime(2021, 5, 31, 23, 59, 59)   # add date and time context of end dates to make it inclusive

        summer1Start = datetime.datetime(2021, 6, 1)
        summer1End = datetime.datetime(2021, 7, 15, 23, 59, 59)
        summer2Start = datetime.datetime(2021, 7, 16)
        summer2End = datetime.datetime(2021, 8, 31, 23, 59, 59) # add date and time context of end dates to make it inclusive

        fall1Start = datetime.datetime(2021, 9, 1)
        fall1End = datetime.datetime(2021, 10, 15, 23, 59, 59)
        fall2Start = datetime.datetime(2021, 10, 16)
        fall2End = datetime.datetime(2021, 11, 30, 23, 59, 59)   # add date and time context of end dates to make it inclusive

        winterStart = datetime.datetime(2021, 12, 1)
        winterEnd = datetime.datetime(2021, 12, 31, 23, 59, 59)
        winter2Start = datetime.datetime(2021, 1, 1)
        winter2End = datetime.datetime(2021, 2, 28, 23, 59, 59) # add date and time context of end dates to make it inclusive

        modelPath = ""
        if (obValue >= springStart) & (obValue <= springEnd):
           modelPath = "spring"
        elif (obValue >= summer1Start) & (obValue <= summer1End):
            modelPath = "summer1"
        elif (obValue >= summer2Start) & (obValue <= summer2End):
            modelPath = "summer2"
        elif (obValue >= fall1Start) & (obValue <= fall1End):
            modelPath = "fall1"
        elif (obValue >= fall2Start) & (obValue <= fall2End):
            modelPath = "fall2"
        elif ((obValue >= winterStart) & (obValue <= winterEnd)) or ((obValue >= winter2Start) and (obValue <= winter2End)):
            modelPath = "winter"
        # print(modelPath)
        return modelPath


if __name__ == "__main__":
    testObj = TestPipeLine("/Users/vignesh/Desktop/Projects/Econet/models/", 
    "/Users/vignesh/Desktop/Projects/Econet/splitDataMod/testEncoded.csv")
    testDf = pd.read_csv(testObj.testFilePath)

    # testDf["Ob"] = pd.to_datetime(testDf["Ob"], infer_datetime_format=True)

    testPredictions = []

    # load models initially
    seasons = ["fall1", "fall2", "summer1", "summer2", "spring", "winter"]
    testObj.loadModels()

    # predict classes
    testPredictions = testObj.predictClasses(testDf)    
    
    # split it by season
    # for i in range(0,testDf.shape[0]):
    #     eachDataPoint = testDf.iloc[[i]]
    #     modelName = testObj.getSeasonValue(eachDataPoint)
    #     modelLoc = open(testObj.basePath + modelName + ".sav", 'rb')
    #     model = pickle.load(modelLoc)
        
    #     eachDataPoint = eachDataPoint.drop(["Ob", "Station"], axis=1)

    #     # Make predictions - note the use of proba
    #     test_pred = model.predict_proba(eachDataPoint)
    #     testPredictions.append(test_pred[:,1])

    pd.DataFrame(testPredictions, columns=['target']).to_csv('/Users/vignesh/Desktop/Projects/Econet/predictions.csv', index=False)

    """
    'measure_gust02', 'measure_gust10''measure_wd02', 'measure_wd10' these 4 features do not exist in the train data they only exist in the test data!!!
    """
