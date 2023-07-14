import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')
            # Define which columns should be one hot encoded
            categorical_cols = ['sex',
                                'on_thyroxine',
                                'query_on_thyroxine',
                                'on_antithyroid_medication',
                                'sick',
                                'pregnant',
                                'thyroid_surgery',
                                'I131_treatment',
                                'query_hypothyroid',
                                'query_hyperthyroid',
                                'lithium',
                                'goitre',
                                'tumor',
                                'hypopituitary',
                                'psych',
                                'TSH_measured',
                                'T3_measured',
                                'TT4_measured',
                                'T4U_measured',
                                'FTI_measured',
                                'TBG_measured'
                                ]
            numerical_cols = ['age','TSH','T3','TT4','T4U','FTI']

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median'))]
            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ohe',OneHotEncoder())]) 
                
            

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)

            ])

            return preprocessor
        
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
            

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            
            # Defining target class
            train_df['class'] = train_df['class'].apply(lambda x:x.split('.')[0])
            train_df["class"] = train_df["class"].map({'negative':'N','compensated hypothyroid':'P','primary hypothyroid':'P','secondary hypothyroid':'P'})
            test_df['class'] = test_df['class'].apply(lambda x:x.split('.')[0])
            test_df["class"] = test_df["class"].map({'negative':'N','compensated hypothyroid':'P','primary hypothyroid':'P','secondary hypothyroid':'P'})


            # deleting features with no information
            del train_df["TBG"]
            del train_df["referral_source"]
            del test_df["TBG"]
            del test_df["referral_source"]
            
            # repalcing '?' with nan
            train_df=train_df.replace({"?":np.NAN})
            test_df=test_df.replace({"?":np.NAN})

            # converting object dtpes to float of continous features
            numerical_cols = ['age','TSH','T3','TT4','T4U','FTI']

            
            for feat in numerical_cols:
                train_df[feat] = train_df[feat].astype('float')
                test_df[feat] = test_df[feat].astype('float')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')    

            logging.info('Obtaining preprocessing object')
            

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'class'
            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[np.array(input_feature_train_arr), np.array(target_feature_train_df)]
            test_arr = np.c_[np.array(input_feature_test_arr), np.array(target_feature_test_df)]

            
            
            

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)  




            
            