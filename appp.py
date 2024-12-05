# import pandas as pd 
# import nltk
# nltk.download('punkt_tab')
# # nltk.download('stopwords')
# # from nltk.corpus import stopwords
# # stop_words = stopwords.words('english')
# # print(stop_words)

# # from nltk.stem.porter import PorterStemmer
# # ps = PorterStemmer()
# # print(ps.stem('Loving'))

# # a = ['Yogesh','chouhAn','Load']
# # s = pd.Series(a)
# # for i in s:
# #     print(i.lower())
# nltk.word_tokenize('yogesh chouhan ')

# import pandas as pd
# s  = 'yogesh'
# print(pd.Series(s))
# import pickle
# import pandas as pd 
# from text_preprocessing import TextPreprocessing
# s = 'ham,Is that seriously how you spell his name?,,,'

# s = pd.Series(s)
# tp = TextPreprocessing()
# a = tp.preprocess_text(s)
# vect_model = pickle.load(open('vect_model.pkl','rb')) # load vectorization model

# inp = vect_model.transform(a)
# print(inp.shape)

