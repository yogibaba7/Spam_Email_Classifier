import  pandas as pd 
import numpy 
import pickle
from data_cleaning import CleanData
from text_preprocessing import TextPreprocessing
from model_training import ModelTraining

# import data
data = pd.read_csv('spam.csv',encoding='latin1')  # Email spam dataset
data1 = pd.read_csv('Youtube-Spam-Dataset.csv')   # Youtube spam comment dataset

# Clean data 
cd = CleanData(data)
clean_data = cd.perform_data_cleaning()
print(clean_data.shape)

# Text Preprocessing
tp = TextPreprocessing()
x = tp.preprocess_text(clean_data['text']) # Text Preprocessing

tp.Vectorization(x) # create vectorization model 

vect_model = pickle.load(open('vect_model.pkl','rb')) # load vectorization model

inp = vect_model.transform(x) # inp -> input
out = clean_data['target']    # out -> output

print(inp.shape,out.shape)

# Model Training
mt = ModelTraining()
mt.train_model(inp,out) # X as input and y as target

# load prediction model 
model = pickle.load(open('model.pkl','rb'))




# -------------------- App Building ----------------------
import streamlit as st 



# App title and configuration
st.set_page_config(page_title="Text Classifier Web App", layout="wide")
st.sidebar.title("Navigation")

# Sidebar navigation options
option = st.sidebar.radio(
    "Select One",
    options=["Home", "Email/SMS Spam Classifier", "YouTube Comment Spam Classifier"]
)

# Function for the Home Page
def home_page():
    st.title("🌟 Text Classifier App 🌟")
    st.header("Welcome to Your Text Classifier App")
    st.subheader("Let's Predict What Your Text Says...")
    st.markdown(
        """
        Hi, I'm Yogesh Chouhan😊, and I'm presenting this text classifier. In today's fast-paced world, we know that not everyone has unlimited time, right? We all want to save time so we can spend it enjoying life to the fullest📄.

        That’s where my project comes in. This application is designed to help you quickly understand your text or determine its importance in just a few moments. If you're looking to save time and focus on what really matters, give it a try!
        """
    )

# Function for Email/SMS Spam Classifier Page
def email_classifier_page():
    st.title("📧 Email/SMS Spam Classifier")
    st.write("This tool allows you to classify whether an email or SMS is spam or not.")
    # Add a text input field for the user to enter text
    user_input = st.text_area("Enter the email/SMS content:")
    if st.button("Classify"):
        # input text preprocess
        user_input = pd.Series(user_input) # Convert into series
        inp = vect_model.transform(user_input)   # apply vectorization
        
        output = model.predict(inp)





        # Dummy result for now
        if output ==0:
            st.write("📋 Your input text has been classified as **Not Spam**. ")
        else:
            st.write("📋 Your input text has been classified as ** Spam**. ")


# Function for YouTube Comment Spam Classifier Page
def youtube_classifier_page():
    st.title("💬 YouTube Comment Spam Classifier")
    st.write("This tool helps you identify spam comments on YouTube.")
    # Add a text input field for user to enter YouTube comment
    user_input = st.text_area("Enter the YouTube comment:")
    if st.button("Classify"):
        # Dummy result for now
        st.write("📋 Your input text has been classified as **Spam**.")

# Render the selected page
if option == "Home":
    home_page()
elif option == "Email/SMS Spam Classifier":
    email_classifier_page()
elif option == "YouTube Comment Spam Classifier":
    youtube_classifier_page()








