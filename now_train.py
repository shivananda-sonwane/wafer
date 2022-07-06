from training import training_data
import threading

class have_to_train:
    def __init__(self,path):
        self.path=path
    def train_do(self):
        t=training_data(self.path)
        t1 = threading.Thread(target=t.do_training())
        t1.setDaemon(True)
        t1.start()
        ans=t.do_training()
        return ans
