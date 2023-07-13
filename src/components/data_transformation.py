import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            features = ['age',
                        'sex',
                        'on thyroxine',
                        'query on thyroxine',
                        'on antithyroid medication',
                        'sick',
                        'pregnant',
                        'thyroid surgery'
                        'I131 treatment',
                        'query hypothyroid',
                        'query hyperthyroid',
                        'lithium',
                        'goitre',
                        'tumor',
                        'hypopituitary',
                        'psych',
                        'TSH measured',
                        'TSH',
                        'T3 measured',
                        'T3',
                        'TT4 measured',
                        'TT4',
                        'T4U measured',
                        'T4U',
                        'FTI measured',
                        'FTI',
                        'TBG measured',
                        'TBG',
                        'referral source',
                        'class']
            
            # feature engineering of target
            train_df['class'] = train_df['class'].apply(lambda x:x.split('.')[0])
            train_df["class"] = train_df["class"].map({'negative':0,'compensated hypothyroid':1,'primary hypothyroid':1,'secondary hypothyroid':1})
            test_df['class'] = test_df['class'].apply(lambda x:x.split('.')[0])
            test_df["class"] = test_df["class"].map({'negative':0,'compensated hypothyroid':1,'primary hypothyroid':1,'secondary hypothyroid':1})


            # deleting features with no information
            del train_df["TBG"]
            del train_df["referral source"]
            del test_df["TBG"]
            del test_df["referral source"]
            
            # repalcing with dummies
            train_df=train_df.replace({"t":1,"f":0,"?":np.NAN,"F":1,"M":0})
            test_df=test_df.replace({"t":1,"f":0,"?":np.NAN,"F":1,"M":0})

            # converting object dtpes to float
            obj_cols = train_df.columns[train_df.dtypes=='object']
            for feat in obj_cols:
                train_df[obj_cols] = train_df[obj_cols].astype('float')
                test_df[obj_cols] = test_df[obj_cols].astype('float')

            # dealing with null values
            imputer = SimpleImputer(strategy='median')
            nan_feat = ['age','TSH','T3','TT4','T4U','FTI']
            for feat in nan_feat:
                train_df[feat] = imputer.fit_transform(train_df[[feat]])
                test_df[feat] = imputer.fit_transform(test_df[[feat]])
            train_df['sex'] = train_df['sex'].fillna(0.0) 
            test_df['sex'] = test_df['sex'].fillna(0.0) 

            target_column_name = 'class'
            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            preprocessing_obj = pd.concat([train_df,test_df],ignore_index=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_df,
                test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)  




            
            