from datetime import datetime
import logging as lg
class logger:
    """
        This is a logger class which will take as an 
        input filename where all logs will be stored and 
        another input will be the messeage the person want to
        add
        Created by Tanmay Chakraborty
        Date:29-Jun-2022
    """
    def __init__(self,file_name):
        self.file_object=file_name
    def log(self,msg):
        with open(self.file_object,"a+") as f:
            f.write(f"Logging timestamp is {datetime.now()} and {msg}")
