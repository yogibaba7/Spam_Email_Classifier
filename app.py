import  pandas as pd 
import numpy 
import pickle
from data_cleaning import CleanData
from text_preprocessing import TextPreprocessing
from model_training import ModelTraining

# import  data
data = pd.read_csv('spam.csv',encoding='latin1')  # Email spam dataset
youtube_comments = pd.read_csv('Youtube-Spam-Dataset.csv')   # Youtube spam comment dataset

# Clean data 
cd = CleanData()
clean_data = cd.perform_data_cleaning(data) # clean email spam dataset
print(clean_data.shape)

clean_youtube_comments = cd.perform_data_cleaning_on_comment(youtube_comments) # cleam youtube comment dataset
print(clean_youtube_comments.shape)

# Text Preprocessing
tp = TextPreprocessing()
x = tp.preprocess_text_for_email(clean_data['text']) # Text Preprocessing on email
x1 = tp.preprocess_text_for_comments(clean_youtube_comments['CONTENT']) # TEXT PREPROCESSING ON COMMENTS


tp.Vectorization_on_email(x) # create vectorization model on email
tp.Vectorization_on_comments(x1) # create vectorization model on comments

vect_model = pickle.load(open('vect_model.pkl','rb')) # load vectorization model
vect_model_for_comments = pickle.load(open('vect_model_for_comments.pkl','rb')) # load vectorization model of comments


inp = vect_model.transform(x) # inp -> input for email
out = clean_data['target']    # out -> output for email 

inp_for_comments = vect_model_for_comments.transform(x1) # inp -> input for comments
out_for_comments = clean_youtube_comments['CLASS']    # out -> output for comments

print(inp.shape,out.shape)
print(inp_for_comments.shape,out_for_comments.shape)

# Model Training
mt = ModelTraining()
mt.train_model(inp,out) # X as input and y as target / (model for email)
mt.train_model_for_comments(inp_for_comments,out_for_comments)



# load prediction model 
model = pickle.load(open('model.pkl','rb')) # for email
model_for_comments = pickle.load(open('model_for_comments.pkl','rb')) # for comments



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
    st.header("Welcome to Text Classifier App")
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

        user_input = pd.Series(user_input) # Convert into series
        inp = vect_model_for_comments.transform(user_input)   # apply vectorization
        
        output = model_for_comments.predict(inp)

        if output ==0:
            st.write("📋 Your input text has been classified as **Not Spam**.")
        else:
            st.write('📋 Your input text has been classified as **Spam**.')

# Render the selected page
if option == "Home":
    home_page()
elif option == "Email/SMS Spam Classifier":
    email_classifier_page()
elif option == "YouTube Comment Spam Classifier":
    youtube_classifier_page()








