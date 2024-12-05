import pandas as pd
import numpy as np 
import pickle


from sklearn.naive_bayes import MultinomialNB

class ModelTraining:

    def train_model(self,input,output):
        mnb = MultinomialNB()
        mnb.fit(input,output)
        pickle.dump(mnb,open('model.pkl','wb'))
