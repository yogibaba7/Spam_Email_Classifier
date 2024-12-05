import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

class CleanData:
    def __init__(self,data):
        self.data = data
    
    def perform_data_cleaning(self):
        # drop Columns : 'Unnamed: 2','Unnamed: 3','Unnamed: 4'
        self.data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)

        # Rename columns v1 -> target , v2-> text
        self.data.rename(columns={'v1':'target','v2':'text'},inplace=True)

        
        # convert target columsn categories as 0,1 : ham -> 0 , spam -> 1
        target = le.fit_transform(self.data['target'])
        self.data['target'] = target

        # drop duplicates
        self.data.drop_duplicates(inplace=True)

        

        return self.data
