from sqlalchemy import null
from app_logger.logger import logger
import json
import os
import shutil
import pandas as pd
import numpy as np
class raw_data_validation:
    """
    This file is doing all the validations in the given data to checking number of
    column and number of rows in each columns as well as nulls and with that
    the data types also from the schema json data.
    Created by Tanmay Chakraborty.
    Date:29-Jun-2022
    """
    def __init__(self,file_object,path):   
        self.file_object=open(file_object,"a+")
        self.path=path
        self.logger=logger(file_object)
        
    def values_from_schema(self):
        try:
            with open("schema_training.json","r") as f:
                self.logger.log("reading done schema training json file| ")
                dict=json.load(f)
                self.logger.log("Loading of json file done!!| ")
            LengthOfDateStampInFile=dict["LengthOfDateStampInFile"]
            LengthOfTimeStampInFile=dict["LengthOfTimeStampInFile"]
            NumberofColumns=dict["NumberofColumns"]
            first_text=dict["SampleFileName"][:dict["SampleFileName"].find("_")]
            return first_text,LengthOfTimeStampInFile,LengthOfDateStampInFile,NumberofColumns
        except FileNotFoundError as e:
            self.logger.log(f"ISsue has happended while reading the json file and the error is {e}")
        except OSError as e:
            self.logger.log(f"ISsue has happended while reading the json file and the error is {e}")
        except Exception as e:
            self.logger.log(f"ISsue has happended while reading the json file and the error is {e}")
    def create_good_row_directory(self):
        try:
            #directory created for training files
            self.logger.log("Training files directory created!!| ")
            text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"Training_files",exist_ok=True)
            #good directory creation
            os.makedirs("/".join(text.split("/")[:4])+"/"+"Training_files/good_data",exist_ok=True)
            self.logger.log("Good directory created!!| ")
        except OSError as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
        except FileNotFoundError as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
        except Exception as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
    def create_bad_row_directory(self):
        try:
            #bad directory creation
            text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            os.makedirs("/".join(text.split("/")[:4])+"/"+"Training_files/bad_data",exist_ok=True)
            self.logger.log("Bad directory created!!| ")
        except OSError as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
        except FileNotFoundError as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
        except Exception as e:
            self.logger.log(f"Error has happened while creating good direcotry and the error is {e}")
     
    def delete_bad_row_directory(self):
        try:
            text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
            #checking directory is avaialable or not
            if not os.path.isdir("/".join(text.split("/")[:4])+"/"+"Training_files/bad_data"):
                self.logger.log("Bad Directory Deletion Done!!")
                os.rmdir("/".join(text.split("/")[:4])+"/"+"Training_files/bad_data")
            else:
                self.logger.log("Directory is not available already")
                pass
        except FileNotFoundError as e:
            self.logger.log(f"Issue has happened while deletion of bad directory and the error is {e}")
        except Exception as e:
            self.logger.log(f"Issue has happened while deletion of bad directory and the error is {e}")

    def delete_good_row_directory(self):
        text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        try:
            #checking directory is avaialable or not
            if not os.path.isdir("/".join(text.split("/")[:4])+"/"+"Training_files/good_data"):
                self.logger.log("Good Directory Deletion Done!!")
                os.rmdir("/".join(text.split("/")[:4])+"/"+"Training_files/good_data")
            else:
                self.logger.log("Directory is not available already")
                pass
        except FileNotFoundError as e:
            self.logger.log(f"Issue has happened while deletion of good directory and the error is {e}")
        except Exception as e:
            self.logger.log(f"Issue has happened while deletion of bad directory and the error is {e}  ")
    def csv_files_total_validation_checking(self,first_text,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        for files in os.listdir(self.path):
            try:
                self.logger.log("CHecking conditions of csv file checking  ")
                self.logger.log(f"text check {files[:files.find('_')].lower()}=={first_text}")
                self.logger.log(f"text check {(files[:files.find('_')].lower()==first_text)}")
                self.logger.log(f"date check {(len(str(files[files.find('_')+1:files.find('_')+9])))}=={LengthOfDateStampInFile}")
                self.logger.log(f"date check {(len(str(files[files.find('_')+1:files.find('_')+9])))==LengthOfDateStampInFile}")
                self.logger.log(f"Condition Check {(files[:files.find('_')].lower()==first_text) and (len(str(files[files.find('_')+1:files.find('_')+9])))==LengthOfDateStampInFile}")
                #it will be checking all the digits of given dates are number or not using for loop
                date_len_validation=[]
                for i in range(len(files[files.find("_")+1:files.find("_")+9])):
                    if files[files.find("_")+1:files.find("_")+9][i].isdigit():
                        date_len_validation.append(True)
                if date_len_validation.count(True)==len(files[files.find("_")+1:files.find("_")+9]):
                    #it will be checking all the digits of given time is number or not using for loop
                    time_len_validation=[]
                    extracted_text=files[files.find("_")+1:]
                    for i in range(len(extracted_text[extracted_text.find("_")+1:extracted_text.find("_")+7])):
                        if extracted_text[extracted_text.find("_")+1:extracted_text.find("_")+7][i].isdigit():
                            time_len_validation.append(True)
                    if time_len_validation.count(True)==len(extracted_text[extracted_text.find("_")+1:extracted_text.find("_")+7]):


                        if (files[:files.find('_')]==first_text) and (len(str(files[files.find("_")+1:files.find("_")+9])))==LengthOfDateStampInFile:
                            self.logger.log(f"{files[:5]}=={first_text} matched")
                            self.logger.log(f"{len(str(files[files.find('_')+1:files.find('_')+9]))}=={LengthOfDateStampInFile} matched")
                            extracted_text=files[files.find("_")+1:]
                            if len(str(extracted_text[extracted_text.find("_")+1:extracted_text.find("_")+7]))== LengthOfTimeStampInFile:
                                self.logger.log(f"{len(str(extracted_text[extracted_text.find('_')+1:extracted_text.find('_')+7]))}=={LengthOfTimeStampInFile} matched")
                                if files[-4:]==".csv":
                                    self.logger.log("Moving good files to good folder  ")
                                    self.logger.log("Reading using pandas ")
                                    df=pd.read_csv(os.path.join(self.path,files))
                                    cols_lists=["Wafer_names"]
                                    for cols in df.columns[1:]:
                                        cols_lists.append(cols)
                                    self.logger.log("Updating columns ")
                                    df.columns=cols_lists
                                    self.logger.log("Column updated ")
                                    df.to_csv(os.path.join("/".join(text.split("/")[:4])+"/"+"Training_files/good_data",files))
                                    self.logger.log("Csv conversion done and added to good folder!! ")
                                    # shutil.copy("E:/ml project/classification project/wafer Fault Detection Project/training_batch_files/"+files,"E:/ml project/classification project/Wafer Fault Detection Project/Training_files/good_data")
                                    self.logger.log(f"{files} moved to good folder  ")
                        else:
                            self.logger.log("Moving bad files to bad folder  ")
                            self.logger.log(f"{files} move to bad folder"  )
                            self.logger.log(f"{files[:5].lower()}=={first_text} not matched  ")
                            self.logger.log(f"{len(str(files[files.find('_')+1:files.find('_')+9]))}=={LengthOfDateStampInFile} not matched  ")
                            shutil.copy(self.path+"/"+files,"/".join(text.split("/")[:4])+"/"+"Training_files/bad_data")
                    else:
                        self.logger.log("Moving bad files to bad folder  ")
                        self.logger.log(f"{files} move to bad folder"  )
                        self.logger.log(f"{files[:5].lower()}=={first_text} not matched  ")
                        self.logger.log(f"{len(str(files[files.find('_')+1:files.find('_')+9]))}=={LengthOfDateStampInFile} not matched  ")
                        shutil.copy(self.path+"/"+files,"/".join(text.split("/")[:4])+"/"+"Training_files/bad_data")      
                else:
                    self.logger.log("Moving bad files to bad folder  ")
                    self.logger.log(f"{files} move to bad folder"  )
                    self.logger.log(f"{files[:5].lower()}=={first_text} not matched  ")
                    self.logger.log(f"{len(str(files[files.find('_')+1:files.find('_')+9]))}=={LengthOfDateStampInFile} not matched  ")
                    shutil.copy(self.path+"/"+files,"/".join(text.split("/")[:4])+"/"+"Training_files/bad_data")
                    
            except Exception as e:
                self.logger.log(f"Error has happened while checking csv file criteria and the error is {e}")

    def column_validaion_withfillna(self):
        """
        This will fill the all blanks rows as null rows and check the column validation as per of dsA
        """
        text=self.path.encode('unicode_escape').decode("ASCII").replace("\\\\","/").replace("\\","/")
        try:
            os.makedirs("/".join(text.split("/")[:4])+"/"+"Train_to_DB",exist_ok=True)
            for files in os.listdir("/".join(text.split("/")[:4])+"/"+"Training_files/good_data"):
                self.logger.log(f"Reading the file {files}")
                df=pd.read_csv(os.path.join("/".join(text.split("/")[:4])+"/"+"Training_files/good_data",files))
                df=df.fillna(10000)
                df.drop("Unnamed: 0",axis=1,inplace=True)
                self.logger.log("Filling blanks with some ambiguous value ")
                #reading the json file
                with open("schema_training.json") as file:
                    dict=json.load(file)
                if (len(df.columns)==592) and (len(dict["ColName"].keys())==592):
                    self.logger.log("length of column checking has been done ")
                    check=[True,True]
                    self.logger.log("Column names checking started!! ")
                    for cols,jsons in zip(df.columns,dict["ColName"].keys()):
                        jsons=jsons.replace(" ","")
                        if cols==jsons:
                            self.logger.log(f"check {cols}=={jsons}")
                            check.append(True)
                    self.logger.log(f"Count of correct columns in {files} is {check.count(True)}")
                    if check.count(True)==592:
                        df.to_csv(os.path.join("/".join(text.split("/")[:4])+"/"+"Train_to_DB",files),index=False)
                        self.logger.log("sending the csv to train_db folder which send that data to database")
                    else:
                        df.to_csv(os.path.join("/".join(text.split("/")[:4])+"/"+"Training_files/bad_data",files),index=False)


        except Exception as e:
            self.logger.log(f"Problem while reading the good data directory and the error is {e}")





            

