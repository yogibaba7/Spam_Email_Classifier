import pandas as pd 
import numpy as np 
import pickle

import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


        

class TextPreprocessing:

    def preprocess_text(self,series):
        l = []
        for text in series:
            # convert into lower case
            text = text.lower()

            # special characters
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

            # tokenization
            text = nltk.word_tokenize(text)

            # remove stop words
            ns_words = []
            for i in text:
                if i not in stop_words:
                    ns_words.append(i)
    


    
            stem_words = []
            for i in ns_words:
                stem_words.append(ps.stem(i))
            text = " ".join(stem_words)

            l.append(text)
        s = pd.Series(l)
        return s 

    def Vectorization(self,series):    
        # Vectorization
        tfidf = TfidfVectorizer(max_features=2000)
        tfidf.fit(series)
        pickle.dump(tfidf,open('vect_model.pkl','wb'))

