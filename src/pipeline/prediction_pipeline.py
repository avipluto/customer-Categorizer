import os
import src.ml.model.s3_estimator
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.constant.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import CustomerException
import pandas as pd
import numpy as np
import sys

import logging
import sys
from pandas import DataFrame
import pandas as pd





class CustomerData:
    def __init__(self):
        pass
        
    def get_input_dataset(self, column_schema:dict, input_data):
        columns = column_schema.keys()
        
        input_dataset = pd.DataFrame([input_data], columns = columns)
        for key, value in column_schema.items():
            input_dataset[key] = input_dataset[key].astype(value)
        
        return input_dataset

    @staticmethod
    def form_input_dataframe(data):
        prediction_config = Prediction_config()
        prediction_schema = prediction_config.__dict__
        column_schema = prediction_schema['prediction_schema']['columns']

        customerData = CustomerData()
        input_dataset = customerData.get_input_dataset(
            column_schema=column_schema,
            input_data=data
        )
        
        return input_dataset
        
        
    


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()

        # NOTE: AWS/S3 isn't configured yet, so instead of loading the model
        # from S3 (via CustomerClusterEstimator), we load it from the local
        # folder that model_pusher.py now saves to. Once AWS is set up, this
        # can be swapped back to the S3-based get_trained_model() below.
        self.local_model_path = os.path.join(os.getcwd(), "saved_models", "model.pkl")
        
    def prepare_input_data(self, input_data:list) -> pd.DataFrame:
        """ 
        method: prepare_input_data 
        
        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data 

        Raises:
            CustomerException

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """
        try:
        
            
            customerDataframe = CustomerData.form_input_dataframe(data = input_data)
            logging.info("customerDatafram has been created")
            return customerDataframe
        except Exception as e:
            raise CustomerException(e,sys)
        
   
        
    
        
    def get_trained_model(self, ModelTrainerConfig = ModelTrainerConfig):
        """
        method: get_trained_model
        
        objective: this method returns the model

        Args:
            ModelTrainerConfig

        Raises:
            CustomerException: 

        Returns:
            model: latest trained model
        """
        try:
            prediction_config = PredictionPipelineConfig()
            model = src.ml.model.s3_estimator.CustomerClusterEstimator(
                bucket_name= prediction_config.model_bucket_name,
                model_path= prediction_config.model_file_name
            )
                
            return model
                
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_local_trained_model(self):
        """
        method: get_local_trained_model

        objective: Loads the trained model from the local saved_models folder
                    (written there by model_pusher.py) instead of S3, since
                    AWS isn't configured yet.

        Returns:
            model: locally saved CustomerSegmentationModel object
        """
        try:
            if not os.path.exists(self.local_model_path):
                raise Exception(
                    f"No locally saved model found at {self.local_model_path}. "
                    f"Please run /train first to generate one."
                )
            return self.utils.load_object(file_path=self.local_model_path)
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def run_pipeline(self, input_data:list):
        
        """
        method: run_pipeline
        
        objective: run_pipeline method runs the whole prediction pipeline.

        Raises:
            CustomerException: 
        """
        try:
            input_dataframe =  self.prepare_input_data(input_data) 
            model = self.get_local_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
            
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_cluster(self, input_data: list):
        """
        Alias for run_pipeline(), matching the method name app.py's
        predict endpoint calls.
        """
        return self.run_pipeline(input_data)