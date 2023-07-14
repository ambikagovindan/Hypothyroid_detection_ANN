import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix    
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],
                                             train_arr[:,-1],
                                             test_arr[:,:-1],
                                             test_arr[:,-1])
            models = {
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForestClassifier':RandomForestClassifier(),
                'LogisticRegression':LogisticRegression()
            }

            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,y_train)
    
                #Make Predictions
                y_pred=model.predict(X_test)
                class_report,cm=evaluate_model(y_test,y_pred)
                print(list(models.keys())[i])
                print(class_report)
                print('confusion matrix')
                print(cm)
    
    
                print('='*35)
                print('\n')

            logging.info('models trained')
            best_model = DecisionTreeClassifier()
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)      