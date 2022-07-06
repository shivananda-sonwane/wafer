import os
import pandas as pd
import numpy as np
import pickle
from app_logger.logger import logger
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
class prediction:
    """
    This class will be used to do prediction of wafers
    """
    def do_prediction(path):
        text=path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        text=text.replace("x0c","f")

        for files in os.listdir("/".join(text.split("/")[:len(text.split("/"))])):
            df=pd.read_csv("/".join(text.split("/")[:len(text.split("/"))])+"/"+files)
            df=df.fillna(0)
            scaler=pickle.load(file=open("/".join(text.split("/")[:text.split("/").index("Training_files")])+"/model/"+"scaler.pkl","rb"))
            prediction1=pickle.load(file=open("/".join(text.split("/")[:text.split("/").index("Training_files")])+"/model/"+"cluster1/logistics_py.pkl","rb"))
            prediction2=pickle.load(file=open("/".join(text.split("/")[:text.split("/").index("Training_files")])+"/model/"+"cluster2/logistics_py.pkl","rb"))
            wafer_names=df["Wafer_names"]
            df.drop("Wafer_names",axis=1,inplace=True)
            if "Good/Bad" in df.columns:
                df.drop("Good/Bad",axis=1,inplace=True)
            else:
                pass
            if "Unnamed: 0" in df.columns:
                df.drop("Unnamed: 0",axis=1,inplace=True)
            else:
                pass
            colnames=df.columns
            df_new=pd.DataFrame(scaler.transform(df),columns=colnames)
            from sklearn.cluster import DBSCAN
            db=DBSCAN(eps=19,algorithm="brute")
            cluster_df=pd.DataFrame(db.fit_predict(df_new))
            cluster_df.columns=["Cluster"]
            df=pd.concat([wafer_names,df_new,cluster_df],axis=1)
            cluster1=df[df["Cluster"]==1]
            cluster1.drop("Cluster",axis=1,inplace=True)
            cluster2=df[df["Cluster"]==-1]
            cluster2.drop("Cluster",axis=1,inplace=True)
            if cluster1.shape[0]!=0:
                cluster1.drop("Wafer_names",axis=1,inplace=True)
                if "Good/Bad" in cluster1.columns:
                    df.drop("Good/Bad",axis=1,inplace=True)
                else:           
                    pass
                colnames=cluster1.columns
                df_new=pd.DataFrame(scaler.transform(cluster1),columns=colnames)
                output=pd.DataFrame()
                output["wafer names"]=wafer_names
                data_prob_y=prediction1.predict_proba(df_new)[:,0]
                for i in range(data_prob_y.shape[0]):
                    if data_prob_y[i]>0.9:
                        output.loc[i,"pred"]=-1
                    else:
                        output.loc[i,"pred"]=1
                output.to_csv("/".join(text.split("/")[:text.split("/").index("good_data")])+"/output.csv")
            elif cluster2.shape[0]!=0:
                cluster2.drop("Wafer_names",axis=1,inplace=True)
                if "Good/Bad" in cluster1.columns:
                    df.drop("Good/Bad",axis=1,inplace=True)
                else:           
                    pass
                colnames=cluster2.columns
                df_new=pd.DataFrame(scaler.transform(cluster2),columns=colnames)
                output=pd.DataFrame()
                output["wafer names"]=wafer_names
                data_prob_y=prediction2.predict_proba(df_new)[:,0]
                for i in range(data_prob_y.shape[0]):
                    if data_prob_y[i]>0.9:
                        output.loc[i,"pred"]=-1
                    else:
                        output.loc[i,"pred"]=1
                output.to_csv("/".join(text.split("/")[:text.split("/").index("good_data")])+"/output.csv")
            else:
                pass
        return f'Prediction done please check your output csv in predicting files folder and the path is {"/".join(text.split("/")[:text.split("/").index("good_data")])+"/output.csv"}'


