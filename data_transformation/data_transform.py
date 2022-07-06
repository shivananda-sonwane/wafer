from app_logger.logger import logger
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle

class data_transformation:
    """
    This is class will help to transform your data for example here you can 
    impute missing values with best imputer and as weelll as you can scale
    your data
    Created By Tanmay Chakraborty
    Date:1-Jul-2022
    """
    def __init__(self,df):
        self.df=df
        self.logger=logger("transformation.txt")    
    def splitting_the_data(self):
        try:
            #extracting x from main data
            df_x=self.df.drop("target",axis=1)
            self.logger.log("Extracting the data for x axis ")
            #extracting y from main data
            df_y=self.df["target"]
            self.logger.log("Extracting the data for y axis ")
            x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=42)
            self.logger.log("Data Splitting has been done!! ")
            data_x=pd.concat([x_train,x_test])
            self.logger.log("X Axis Concatenation Done!! ")
            data_y=pd.concat([y_train,y_test])
            self.logger.log("Y Axis concatenation Done! ")
            wafer_names=data_x["Wafer_names"].reset_index(drop=True)
            self.logger.log("Wafer Names columns created")
            data_x.drop(["Wafer_names"],axis=1,inplace=True)
            colnames=data_x.columns
            data_x=data_x.replace(10000,np.nan)
            data_x=data_x.reset_index(drop=True)
            data_y=data_y.reset_index(drop=True)
            return data_x,data_y,colnames,wafer_names
        except Exception as e:
            self.logger.log(f"Error has happened while doing the splitting of the data and the error is {e}")
            return "Error has happened while doing the splitting of the data"
        
    def impute_nulls(self,data_x,data_y,colnames,wafer_names):
        try:
            imputer=KNNImputer(n_neighbors=5,weights="distance")
            self.logger.log("Applying KNN Imputer ")
            data_x=imputer.fit_transform(data_x)
            self.logger.log("Fitting the data in imputer ")
            df_new=pd.DataFrame(data_x,columns=colnames)
            self.logger.log("New Data Created!! ")
            return df_new
        except Exception as e:
            self.logger.log(f"Issue has happened while doing the imputation of nulls the error is {e} ")
            return "There is something wrong during imputation!"
    
    def feature_scaler(self,data,colnames,path):
        try:
            text=path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            self.logger.log("Inside feature scaler class!! ")
            scaler=StandardScaler()
            self.logger.log("Standard scaler object has been created")
            df_trans=pd.DataFrame(scaler.fit_transform(data),columns=colnames)
            self.logger.log("Scaler has been fitted")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model",exist_ok=True)
            pickle.dump(scaler,file=open("/".join(text.split("/")[:4])+"/"+"model/scaler.pkl","wb"))
            return scaler
        except Exception as e:
            self.logger.log(f"Isssue has happened while doing feature scaling!! and the error is {e}")
            return "Something error occurred in feature scaler class"

    def scaled_data(self,data,colnames,scaler,wafer_names,data_y,path):
        text=path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        try:
            self.logger.log("Inside scaled data class")
            self.logger.log("calling feature scaler object done")
            df_trans=pd.DataFrame(scaler.fit_transform(data),columns=colnames)
            df_trans=pd.concat([wafer_names,df_trans,data_y],axis=1)
            self.logger.log("Dataframe created!! ")
            df_trans.to_csv("/".join(text.split("/")[:4])+"/"+"transformed_data.csv",index=False)
            self.logger.log("transformed csv created!! ")
            return "transformed_data has been ready for training"
        except Exception as e:
            self.logger.log("Issue has happened in scaled_data class! ")
            return "Something wrong in scaled data class"



    
