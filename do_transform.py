from app_logger.logger import logger
from data_transformation.data_transform import data_transformation

class do_transformation:
    def __init__(self,df,path):
        self.transform=data_transformation(df)
        self.df=df
        self.path=path
        self.logger=logger("do_transformation_data.txt")
    def transform_data(self):
        try:
            text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            self.logger.log("Inside transformed data ")
            data_x,data_y,colnames,wafer_names=self.transform.splitting_the_data()
            self.logger.log("Data Splitting has been done!! ")
            df_new=self.transform.impute_nulls(data_x,data_y,colnames,wafer_names)
            self.logger.log("Imputing nulls has been done!! ")
            scaler=self.transform.feature_scaler(df_new,colnames,self.path)
            self.logger.log("Feature scaling done!! ")
            self.transform.scaled_data(df_new,colnames,scaler,wafer_names,data_y,self.path)
            return "transformation done"
        except Exception as e:
            self.logger.log(f"Error while doing the transformation and the errror is {e}")
            return "Something error while transform data!!"
      



