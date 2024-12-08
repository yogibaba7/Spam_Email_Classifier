import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

class CleanData:

    
    def perform_data_cleaning(self,data):
        # drop Columns : 'Unnamed: 2','Unnamed: 3','Unnamed: 4'
        data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)

        # Rename columns v1 -> target , v2-> text
        data.rename(columns={'v1':'target','v2':'text'},inplace=True)

        
        # convert target columsn categories as 0,1 : ham -> 0 , spam -> 1
        target = le.fit_transform(data['target'])
        data['target'] = target

        # drop duplicates
        data.drop_duplicates(inplace=True)

        

        return data

    def perform_data_cleaning_on_comment(self,data):
        # remove 'COMMENT_ID','AUTHOR','DATE','VIDEO_NAME' columns
        data.drop(columns = ['COMMENT_ID','AUTHOR','DATE','VIDEO_NAME'],inplace=True)
        # Drop Duplicate Values
        data.drop_duplicates(inplace=True)

        return data

