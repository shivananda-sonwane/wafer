from matplotlib.style import use
from raw_data_validation_py import raw_data_validation
from app_logger.logger import logger
class train_validation:
    """
    Training validation is going on each csv files using different other classes
    as specified.
    Created by Tanmay Chakraborty.
    Date:29-Jun-2022
    """
    def __init__(self,file_log,path):
        self.file_log=open(file_log,"a+")
        self.raw_data=raw_data_validation("raw_data_validation.txt",path)
        self.path=path
        self.logger=logger(file_log)
    def training_validation(self):
        try:
            #validation starting log
            self.logger.log("Data Validation has started!!| ")
            #taking outputs from raw_data_validation from values_from_schema to
            first_text,LengthOfTimeStampInFile,LengthOfDateStampInFile,NumberofColumns=self.raw_data.values_from_schema()
            self.logger.log("Detection of schema done!!| ")
            #creating good row directory
            self.raw_data.create_good_row_directory()
            self.logger.log("Good row directory created!!| ")
            #creating bad row directory
            self.raw_data.create_bad_row_directory()
            self.logger.log("Bad row directory created!!| ")

            #using the validation of csv with the schema
            self.logger.log("Validation of all csv with schema has started| ")
            self.raw_data.csv_files_total_validation_checking(first_text,LengthOfDateStampInFile,LengthOfTimeStampInFile)

            #filling blanks with nulls  and column check
            self.raw_data.column_validaion_withfillna()

        except Exception as e:
            self.logger.log(f"error has happened while doing the training validation and the error is {e}")




