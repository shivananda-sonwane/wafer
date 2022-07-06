from app_logger.logger import logger
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
from data_transformation.data_transform import data_transformation
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

class training_data:
    def __init__(self,path):
        self.path=path
    def do_training(self):
        text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        df=pd.read_csv(os.path.join("/".join(text.split("/")[:4]),"transformed_data.csv"))
        transform=data_transformation(df)
        logg=logger("training_log.txt")
        data_x,data_y,colnames,wafer_names=transform.splitting_the_data()
        try:
            os1=RandomOverSampler(0.75)
            logg.log("os1 created!!")
            #creating clusters 
            # calling DBSCAN
            db=DBSCAN(eps=19,algorithm="brute")
            #creating a dataframe called clustered df which will be stored all outputs
            cluster_df=pd.DataFrame(db.fit_predict(data_x))
            cluster_df.columns=["Cluster"]
            clustered_df=pd.concat([wafer_names,data_x,data_y,cluster_df],axis=1)
            #seperating clusters
            #first cluster
            cluster_1=clustered_df[clustered_df["Cluster"]==0]
            #second cluster
            cluster_2=clustered_df[clustered_df["Cluster"]==-1]
            #dropping Cluster column from first clustered data
            cluster_1.drop("Cluster",axis=1,inplace=True)
            #dropping Cluster column from second clustered data
            cluster_2.drop("Cluster",axis=1,inplace=True)
            #separate x and y in cluster 1
            cluster_1_x=cluster_1.drop("target",axis=1)
            cluster_1_y=cluster_1["target"]
            #separate x and y in cluster 2
            cluster_2_x=cluster_2.drop("target",axis=1)
            cluster_2_y=cluster_2["target"]
            #cluster 1 entire training
            # x_train,x_test,y_train,y_test=train_test_split(cluster_1_x,cluster_1_y,test_size=0.2,random_state=42)
            # # data_x=x_train
            # # data_y=y_train
            # data_x=pd.concat([x_train,x_test])
            # data_y=pd.concat([y_train,y_test])
            data_x=cluster_1_x
            data_y=cluster_1_y
            wafer_names=data_x["Wafer_names"]
            data_x.drop(["Wafer_names"],axis=1,inplace=True)
            colnames=data_x.columns
            #applying logistics regression
            from sklearn.linear_model import LogisticRegression
            log=LogisticRegression()
            # params={"penalty":['l1', 'l2', 'elasticnet'],"tol":[1e-4,1e-5,1e-6,1e-7],"solver":['liblinear', 'sag', 'saga'],"max_iter":[100,200,300,400]}
            # grid=GridSearchCV(estimator=log,param_grid=params)
            # grid.fit(data_x,data_y)
            X_train_ns,y_train_ns=os1.fit_resample(data_x,data_y)
            log=LogisticRegression(penalty="l1",max_iter=400,solver="saga")
            log.fit(X_train_ns,y_train_ns)
            data_pred_y=log.predict(X_train_ns)
            data_pred_prob=log.predict_proba(X_train_ns)
            #metrics function
            def metrics_1(y_true,y_pred):
                tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
                accuracy=(tp+tn)/(tp+fp+tn+fn)
                precision=tp/(tp+fp)
                recall=tp/(tp+fn)
                specificty=tn/(tn+fp)
                f1=2*(precision*recall)/(precision+recall)
                res={"precision":round(precision,2),"accuracy":round(accuracy,2),"recall":round(recall,2),"specificity":round(specificty,2),"f1":round(f1,2)}
                return res
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_logistics.txt","w+") as f:
                f.write("Logistics Regression metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of Logistics Regression")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
    
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/logistics_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(log,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/logistics_py.pkl",'wb'))
            #applying svc
            # svc=SVC()
            # params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"degree":[1,2,3],"gamma":['scale', 'auto'],"tol":[1e-1,1e-2,1e-3,1e-4,1e-5],"decision_function_shape":['ovo', 'ovr'] }
            # grid=GridSearchCV(estimator=svc,param_grid=params)
            # grid.fit(data_x,data_y)
            # grid.best_params_
            svc=SVC(kernel="poly",degree=1,gamma="scale",tol=0.1,decision_function_shape="ovo",probability=True,random_state=42)
            svc.fit(X_train_ns,y_train_ns)
            data_pred_y=svc.predict(X_train_ns)
            data_pred_prob=svc.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_svc.txt","w+") as f:
                f.write("svc metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of svc")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
    
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/svc_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(svc,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/svc_py.pkl",'wb'))
            #applying Decision Tree
            dr=DecisionTreeClassifier()
            # params={"criterion":['gini', 'entropy', 'log_loss'],"splitter":['best', 'random'],"max_depth":range(10,50),"max_features":['auto', 'sqrt', 'log2']}
            # grid=GridSearchCV(estimator=dr,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            dr=DecisionTreeClassifier(criterion="gini",max_features="sqrt",ccp_alpha=0.031,splitter="best",random_state=42)
            dr.fit(X_train_ns,y_train_ns)
            data_pred_y=dr.predict(X_train_ns)
            data_pred_prob=dr.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_dr.txt","w+") as f:
                f.write("decison tree metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of svc")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/dr_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(log,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/dr_py.pkl",'wb'))
            #Gradient boosting classifier
            gr=GradientBoostingClassifier(n_estimators=100)
            # params={"loss":['log_loss', 'deviance', 'exponential'],"learning_rate":[0.1,0.001,0.0001],"criterion":['friedman_mse', 'squared_error', 'mse'],"max_features":['auto', 'sqrt', 'log2']}
            # grid=GridSearchCV(estimator=gr,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            gr=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,criterion="mse",loss="log_loss",max_features="log2")
            gr.fit(X_train_ns,y_train_ns)
            data_pred_y=gr.predict(X_train_ns)
            data_pred_prob=gr.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_gr.txt","w+") as f:
                f.write("gradient boosting metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of gr")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/gr_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(gr,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/gr_py.pkl",'wb'))
            #applying knn
            knn=KNeighborsClassifier()
            params={"n_neighbors":[2,3,4,5,6],"weights":['uniform', 'distance'],"algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']}
            # grid=GridSearchCV(estimator=knn,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            knn=KNeighborsClassifier(n_neighbors=7,weights="uniform",algorithm="ball_tree")
            knn.fit(X_train_ns,y_train_ns)
            data_pred_y=knn.predict(X_train_ns)
            data_pred_prob=knn.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_knn.txt","w+") as f:
                f.write("knn metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of knn")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/knn_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(knn,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/knn_py.pkl",'wb'))
            #xgboost
            y_train_ns_xg=y_train_ns.replace(-1,0)
            xg=XGBClassifier()
            # params={"base_score":[0.4],"booster":["gbtree","gblinear","dart"],"learning_rate":[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
            # grid=GridSearchCV(xg,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            xg=XGBClassifier(base_score=0.7,booster="gbtree",learning_rate=0.1)
            xg.fit(X_train_ns,y_train_ns_xg)
            data_pred_y=xg.predict(X_train_ns)
            data_pred_prob=xg.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns_xg,data_pred_y)
            logg.log("roc auc calculated")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_xg.txt","w+") as f:
                f.write("xgboost metrics")
                logg.log("writting has been started!!")
                f.write(str(metrics_1(y_train_ns_xg,data_pred_y)))
                logg.log("metrics added")
                f.write(f"auc score is {auc}")
                logg.log("auc calculated")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns_xg,data_pred_y)
            logg.log("roc curve calculated!!")
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of xg")
            threshold=[i/10 for i in range(1,10)]
            logg.log("threshold list created")
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            logg.log("threshold df created")
            df["thres"]=data_pred_prob[:,0]
            logg.log("thres col created")
            for i in range(0,df.shape[0]):
                logg.log(f"inside {i} loop")
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=0
                    logg.log("threshold cond check done")
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            logg.log("performance metrics df created")
            for i in threshold:
                logg.log(f"{i} threshold")
                data_pred_y=df[str(i)]
                logg.log(f"{i} data pred created")
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns_xg,np.array(data_pred_y))
                logg.log(f"{i} metrics done")
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
            logg.log("precision recall all done")
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/xg_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(xg,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/xg_py.pkl",'wb'))
            # Random Forest classifier
            rr=RandomForestClassifier(criterion="gini",max_features="sqrt",ccp_alpha=0.031,random_state=42)
            rr.fit(X_train_ns,y_train_ns)
            data_pred_y=rr.predict(X_train_ns)
            data_pred_prob=rr.predict_proba(X_train_ns)
            logg.log("predict probability of rr done")
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_rr.txt","w+") as f:
                logg.log("accuracy file rr opened successfully")
                f.write("random forest tree metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            logg.log("accruracy rr writing done")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of rr")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/rr_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster1",exist_ok=True)
            pickle.dump(rr,open("/".join(text.split("/")[:4])+"/"+"model/cluster1/rr_py.pkl",'wb'))
            #cluster2 entire training
            data_x=cluster_2_x
            data_y=cluster_2_y
            wafer_names=data_x["Wafer_names"]
            data_x.drop(["Wafer_names"],axis=1,inplace=True)
            colnames=data_x.columns
            #applying logistics regression
            from sklearn.linear_model import LogisticRegression
            log=LogisticRegression()
            # params={"penalty":['l1', 'l2', 'elasticnet'],"tol":[1e-4,1e-5,1e-6,1e-7],"solver":['liblinear', 'sag', 'saga'],"max_iter":[100,200,300,400]}
            # grid=GridSearchCV(estimator=log,param_grid=params)
            # grid.fit(data_x,data_y)
            X_train_ns,y_train_ns=os1.fit_resample(data_x,data_y)
            log=LogisticRegression(penalty="l1",max_iter=400,solver="saga")
            log.fit(X_train_ns,y_train_ns)
            data_pred_y=log.predict(X_train_ns)
            data_pred_prob=log.predict_proba(X_train_ns)
            #metrics function
            def metrics_1(y_true,y_pred):
                tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
                accuracy=(tp+tn)/(tp+fp+tn+fn)
                precision=tp/(tp+fp)
                recall=tp/(tp+fn)
                specificty=tn/(tn+fp)
                f1=2*(precision*recall)/(precision+recall)
                res={"precision":round(precision,2),"accuracy":round(accuracy,2),"recall":round(recall,2),"specificity":round(specificty,2),"f1":round(f1,2)}
                return res
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_logistics_c2.txt","w+") as f:
                f.write("Logistics Regression metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of Logistics Regression")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/logistics_perf_c2.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(log,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/logistics_py.pkl",'wb'))
            #applying svc
            svc=SVC()
            params={"kernel":['linear', 'poly', 'rbf', 'sigmoid'],"degree":[1,2,3],"gamma":['scale', 'auto'],"tol":[1e-1,1e-2,1e-3,1e-4,1e-5],"decision_function_shape":['ovo', 'ovr'] }
            # grid=GridSearchCV(estimator=svc,param_grid=params)
            # grid.fit(data_x,data_y)
            # grid.best_params_
            svc=SVC(kernel="poly",degree=1,gamma="scale",tol=0.1,decision_function_shape="ovo",probability=True,random_state=42)
            svc.fit(X_train_ns,y_train_ns)
            data_pred_y=svc.predict(X_train_ns)
            data_pred_prob=svc.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_svc_c2.txt","w+") as f:
                f.write("svc metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of svc")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/svc_perf_c2.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(svc,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/svc_py.pkl",'wb'))
            #applying Decision Tree
            dr=DecisionTreeClassifier()
            # params={"criterion":['gini', 'entropy', 'log_loss'],"splitter":['best', 'random'],"max_depth":range(10,50),"max_features":['auto', 'sqrt', 'log2']}
            # grid=GridSearchCV(estimator=dr,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            dr=DecisionTreeClassifier(criterion="gini",max_features="sqrt",ccp_alpha=0.031,splitter="best",random_state=42)
            dr.fit(X_train_ns,y_train_ns)
            data_pred_y=dr.predict(X_train_ns)
            data_pred_prob=dr.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_dr_c2.txt","w+") as f:
                f.write("decison tree metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of svc")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/dr_perf_c2.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(dr,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/dr_py_c2.pkl",'wb'))
            #Gradient boosting classifier
            gr=GradientBoostingClassifier(n_estimators=100)
            params={"loss":['log_loss', 'deviance', 'exponential'],"learning_rate":[0.1,0.001,0.0001],"criterion":['friedman_mse', 'squared_error', 'mse'],"max_features":['auto', 'sqrt', 'log2']}
            # grid=GridSearchCV(estimator=gr,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            gr=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,criterion="mse",loss="log_loss",max_features="log2")
            gr.fit(X_train_ns,y_train_ns)
            data_pred_y=gr.predict(X_train_ns)
            data_pred_prob=gr.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_gr_c2.txt","w+") as f:
                f.write("gradient boosting metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of gr")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/gr_perf_c2.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(log,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/gr_py.pkl",'wb'))
            #applying knn
            knn=KNeighborsClassifier()
            params={"n_neighbors":[2,3,4,5,6],"weights":['uniform', 'distance'],"algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']}
            # grid=GridSearchCV(estimator=knn,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            knn=KNeighborsClassifier(n_neighbors=7,weights="uniform",algorithm="ball_tree")
            knn.fit(X_train_ns,y_train_ns)
            data_pred_y=knn.predict(X_train_ns)
            data_pred_prob=knn.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_knn_c2.txt","w+") as f:
                f.write("knn metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of knn")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/knn_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(knn,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/knn_py.pkl",'wb'))
            #xgboost
            y_train_ns_xg1=y_train_ns.replace(-1,0)
            xg=XGBClassifier()
            params={"base_score":[0.4],"booster":["gbtree","gblinear","dart"],"learning_rate":[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
            # grid=GridSearchCV(xg,param_grid=params)
            # grid.fit(X_train_ns,y_train_ns)
            # grid.best_params_
            xg=XGBClassifier(base_score=0.7,booster="gbtree",learning_rate=0.1)
            xg.fit(X_train_ns,y_train_ns_xg1)
            data_pred_y=xg.predict(X_train_ns)
            data_pred_prob=xg.predict_proba(X_train_ns)
            auc=roc_auc_score(y_train_ns_xg1,data_pred_y)
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_xg.txt","w+") as f:
                f.write("xgboost metrics")
                f.write(str(metrics_1(y_train_ns_xg1,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns_xg1,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of xg")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=0
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns_xg1,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
        
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/xg_perf.csv")
            # pickle files for model
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            pickle.dump(xg,open("E:/ml project/classification project/Wafer Fault Detection Project/model/cluster2/xg_py.pkl",'wb'))
            # Random Forest classifier
            logg.log("inside rr")
            rr=RandomForestClassifier(criterion="gini",max_features="sqrt",ccp_alpha=0.031,random_state=42)
            logg.log("rr called")
            rr.fit(X_train_ns,y_train_ns)
            logg.log("rr fitted")
            data_pred_y=rr.predict(X_train_ns)
            logg.log("rr predicted")
            data_prob_y=rr.predict_proba(X_train_ns)
            logg.log("prob predicted")
            auc=roc_auc_score(y_train_ns,data_pred_y)
            logg.log("roc auc done")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"metrics_folder",exist_ok=True)
            with open("/".join(text.split("/")[:4])+"/"+"metrics_folder/accuracy_rr_c2.txt","w+") as f:
                f.write("random forest tree metrics")
                f.write(str(metrics_1(y_train_ns,data_pred_y)))
                f.write(f"auc score is {auc}")
                f.write("---END---")
            fpr,tpr,thresholds=roc_curve(y_train_ns,data_pred_y)
            plt.plot(fpr, tpr, color='orange', label='ROC')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            # plt.show()
            plt.savefig("ROC of rr")
            threshold=[i/10 for i in range(1,10)]
            df=pd.DataFrame(columns=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            df["thres"]=data_pred_prob[:,0]
            for i in range(0,df.shape[0]):
                for thres in threshold:
                    if df.loc[i,"thres"]>thres:
                        df.loc[i,str(thres)]=1
                    else:
                        df.loc[i,str(thres)]=-1
            df
            #calculated metrics
            performance_metrics=pd.DataFrame(columns=["precision","accuracy","recall","specificity","f1"],index=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
            for i in threshold:
                data_pred_y=df[str(i)]
                data_pred_y=data_pred_y.astype("int")
                answer=metrics_1(y_train_ns,np.array(data_pred_y))
                performance_metrics.loc[str(i),"precision"]=answer["precision"]
                performance_metrics.loc[str(i),"accuracy"]=answer["accuracy"]
                performance_metrics.loc[str(i),"recall"]=answer["recall"]
                performance_metrics.loc[str(i),"specificity"]=answer["specificity"]
                performance_metrics.loc[str(i),"f1"]=answer["f1"]
            logg.log("before perf")
            performance_metrics.to_csv("/".join(text.split("/")[:4])+"/"+"metrics_folder/rr_perf.csv")
            # pickle files for model
            logg.log("dir created")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"model/cluster2",exist_ok=True)
            logg.log("dumped")
            pickle.dump(rr,open("/".join(text.split("/")[:4])+"/"+"model/cluster2/rr_py.pkl",'wb'))
            return "Training Done!!"
        except Exception as e:
            return f"Error while training error is {e}"



        











        
        

        














