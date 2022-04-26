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
        self.summer1Model = None
        self.summer2Model = None
        self.fall1Model = None
        self.fall2Model = None
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


    def splitTestDf(self, myTestDfX):

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

        winter1Start = datetime.datetime(2021, 12, 1)
        winter1End = datetime.datetime(2021, 12, 31, 23, 59, 59)
        winter2Start = datetime.datetime(2021, 1, 1)
        winter2End = datetime.datetime(2021, 2, 28, 23, 59, 59) # add date and time context of end dates to make it inclusive


        springDf = myTestDfX[(myTestDfX["Ob"] >= springStart) & (myTestDfX["Ob"] <= springEnd)]

        summer1Df = myTestDfX[(myTestDfX["Ob"] >= summer1Start) & (myTestDfX["Ob"] <= summer1End)]
        summer2Df = myTestDfX[(myTestDfX["Ob"] >= summer2Start) & (myTestDfX["Ob"] <= summer2End)]

        fall1Df = myTestDfX[(myTestDfX["Ob"] >= fall1Start) & (myTestDfX["Ob"] <= fall1End)]
        fall2Df = myTestDfX[(myTestDfX["Ob"] >= fall2Start) & (myTestDfX["Ob"] <= fall2End)]

        winter1Df = myTestDfX[(myTestDfX["Ob"] >= winter1Start) & (myTestDfX["Ob"] <= winter1End)]
        winter2Df = myTestDfX[(myTestDfX["Ob"] >= winter2Start) & (myTestDfX["Ob"] <= winter2End)]
        winterDf = pd.concat([winter1Df, winter2Df])

        return springDf, summer1Df, summer2Df, fall1Df, fall2Df, winterDf
    

    def predictClasses(self, dataFrame):
        springDf, summer1Df, summer2Df, fall1Df, fall2Df, winterDf = self.splitTestDf(dataFrame)
        # return

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

        winterDf = winterDf.drop(["Ob", "Station"], axis = 1)
        winterPredict = self.winterModel.predict_proba(winterDf)
        winterPredictDf = pd.DataFrame(winterPredict[:,1], index=winterDf.index)

        concatenatedDf = pd.concat([summer1PredictDf, summer2PredictDf, fall1PredictDf, fall2PredictDf, winterPredictDf, springPredictDf], axis = 0)
        concatenatedDf = concatenatedDf.sort_index(ascending=True)
        # concatenatedDf.to_csv("predictedValues.csv")
        print(concatenatedDf.head())
        # print(summerDf.head())
        # print(fallDf.head())pyt
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
    modelPath1 = "C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/models/"
    testFilePath1 = "C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/testData/testEncoded.csv"
    
    testObj = TestPipeLine(modelPath1, testFilePath1)
    testDf = pd.read_csv(testObj.testFilePath)

    testDf["Ob"] = pd.to_datetime(testDf["Ob"], infer_datetime_format=True)

    # load models initially
    seasons = ["fall1", "fall2", "summer1", "summer2", "spring", "winter"]
    testObj.loadModels()

    # predict classes
    testPredictions = testObj.predictClasses(testDf)
    print(testPredictions.shape)
    testPredictions.columns = ["target"]

    # pd.DataFrame(testPredictions, columns=['target']).to_csv('C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/predictions.csv', index=False)
    testPredictions.to_csv('C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/predictions.csv', index=False)

    """
    'measure_gust02', 'measure_gust10''measure_wd02', 'measure_wd10' these 4 features do not exist in the train data they only exist in the test data!!!
    """
