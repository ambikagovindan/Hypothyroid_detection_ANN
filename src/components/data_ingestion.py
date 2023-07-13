import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
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
                        'class'

            ]
            df1 = pd.read_csv("D:/Hypothyroid_detection/notebook/DATA/allhypo.data",names=features)
            df2 = pd.read_csv("D:/Hypothyroid_detection/notebook/DATA/allhypo.test",names=features)
            df = pd.concat([df1,df2],ignore_index=True)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
'''       
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
'''
if __name__=='__main__':
    obj = DataIngestion() 
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)    
