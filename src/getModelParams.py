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

class LoadModels():
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
    
    def getParams(self):
        print("Spring Model", self.springModel.best_params_)
        print("summer1 Model", self.summer1Model.best_params_)
        print("summer2 Model", self.summer2Model.best_params_)
        print("fall1 Model", self.fall1Model.best_params_)
        print("fall2 Model", self.fall2Model.best_params_)
        print("winter Model", self.winterModel.best_params_)


if __name__ == "__main__":
    modelPath1 = "C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/models/"
    testFilePath1 = "C:/Users/ayrisbud/Downloads/aldaPipeline/final/Econet/testData/testEncoded.csv"
    loadObj = LoadModels(modelPath1, testFilePath1)
    loadObj.loadModels()
    loadObj.getParams()