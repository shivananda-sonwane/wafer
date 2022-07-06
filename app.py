
from wsgiref import simple_server
from flask import Flask,Blueprint,request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from app_logger.logger import logger
from train_validation import train_validation
from training_db import train_data_send_to_database
from do_transform import do_transformation
from training import training_data 
from now_train import have_to_train
from pred_validation import pred_validation
from prediction import prediction
import warnings
warnings.filterwarnings("ignore")

app=Flask(__name__,template_folder="templates")

@app.route("/",methods=["GET","POST"])
def home():
    return render_template("index.html")


@app.route("/train",methods=["POST"])
def train_route():
    #getting the path from the client
    path=request.form.get("paths")
    #sending the path for training validation
    train_data=train_validation("training_validation.txt",path)
    train_data.training_validation()
    #sending data to database after validation
    db=train_data_send_to_database(host="localhost",user="root",passwd="holyshit1234@",path=path)
    db.send_data_todb()
    #getting the data
    csv_data=db.get_data()
    #transformation of csv data
    trans=do_transformation(csv_data,path)
    ans=trans.transform_data()
    train=have_to_train(path)
    ans=train.train_do()
    return ans

@app.route("/predict",methods=["POST"])
def predict_route():
    #getting the path from the client for prediction
    path=request.form.get("pred_folder")
    #sending the path for predicted folder data validation
    pred_data=pred_validation("prediction_validation.txt",path)
    pred_data.prediction_validation()
    pred=prediction
    ans=pred.do_prediction(path)
    return ans










if __name__=="__main__":
    app.run(debug=True)








