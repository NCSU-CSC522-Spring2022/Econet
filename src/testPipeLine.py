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
        self.springModel = None
        self.fallModel = None
        self.summerModel = None
        self.winterModel = None
    
    def loadModels(self, modelPath):
        # print(self.modelFileList)
        # for modelPath in self.modelFileList:
        print(modelPath)
        if "spring" in modelPath:
            springModelLoc = open(modelPath, 'rb')
            self.springModel = pickle.load(springModelLoc)
        if "summer1" in modelPath:
            summerModelLoc = open(modelPath, 'rb')
            self.summerModel = pickle.load(summerModelLoc)
        if "fall" in modelPath:
            fallModelLoc = open(modelPath, 'rb')
            self.fallModel = pickle.load(fallModelLoc)
        if "winter" in modelPath:
            winterModelLoc = open(modelPath, 'rb')
            self.winterModel = pickle.load(winterModelLoc)
        
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
        springDf, summerDf, fallDf, winterDf = self.splitTestDf(dataFrame)
        springPredict = self.springModel.predict(springDf)
        springPredictDf = pd.DataFrame(springPredict, index=springDf.index)

        summerPredict = self.summerModel.predict(summerDf)
        summerPredictDf = pd.DataFrame(summerPredict, index=summerDf.index)

        fallPredict = self.fallModel.predict(fallDf)
        fallPredictDf = pd.DataFrame(fallPredict, index=fallDf.index)

        winterPredict = self.winterModel.predict(winterDf)
        winterPredictDf = pd.DataFrame(winterPredict, index=winterDf.index)

        concatenatedDf = springPredictDf.concat([summerPredictDf, fallPredictDf, winterPredictDf], axis = 0)

        concatenatedDf.to_csv("predictedValues.csv")
        # print(springDf.head())
        # print(summerDf.head())
        # print(fallDf.head())
        # print(winterDf.head())


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
    testObj = TestPipeLine("/Users/vignesh/Desktop/Projects/Econet/models/", "/Users/vignesh/Desktop/Projects/Econet/splitDataMod/testEncoded.csv")
    testDf = pd.read_csv(testObj.testFilePath)

    # testDf["Ob"] = pd.to_datetime(testDf["Ob"], infer_datetime_format=True)

    testPredictions = []
    
    # split it by season
    for i in range(0,testDf.shape[0]):
        eachDataPoint = testDf.iloc[[i]]
        modelName = testObj.getSeasonValue(eachDataPoint)
        modelLoc = open(testObj.basePath + modelName + ".sav", 'rb')
        model = pickle.load(modelLoc)
        
        eachDataPoint = eachDataPoint.drop(["Ob", "Station"], axis=1)

        # Make predictions - note the use of proba
        test_pred = model.predict_proba(eachDataPoint)
        testPredictions.append(test_pred[:,1])

    pd.DataFrame(testPredictions, columns=['target']).to_csv('predictions.csv', index=False)

    """
    'measure_gust02', 'measure_gust10''measure_wd02', 'measure_wd10' these 4 features do not exist in the train data they only exist in the test data!!!
    """
