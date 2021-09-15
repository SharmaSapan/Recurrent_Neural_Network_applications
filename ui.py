# Streamlit Website Implementation to test the RNN Model

# Streamlit Visual Component for the Model Demo
# INSTRUCTIONS TO RUN LOCALLY:
# 1. RUN THE FOLLOWING COMMAND TO INSTALL STREAMLIT (if not already installed): pip install streamlit
# 2. RUN THE FOLLOWING COMMAND  on the terminal TO RUN THIS FILE USING STREAMLIT:    streamlit run ui.py
# 3. In browser then enter text in input box and then its respective analyse sentiment button

# *NOTE- If you get the following error at STEP 2: AttributeError:              module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
#        Run the following command to upgrade a common missing component:       pip install --upgrade protobuf

#
# You can now view your Streamlit app in your browser if it does not pop up it will give url like below.
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.137:8501


import os
import time

import streamlit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow GPU info verbose
import tensorflow as tf
import numpy as np
import io
import regex as re

# Checks input parameters (helps with debug)
@streamlit.cache(allow_output_mutation=True)

class dashboard:

    # Display a Page Title
    def pageTitle(self, title):
        return streamlit.title(title)

    # Display Input Text Section on Page
    def getInputText(self, prompt):
        inputLanguageText = streamlit.text_input(prompt)
        if inputLanguageText == False:
            return
        else:
            return inputLanguageText

    # Display a Subtitle on Page
    def pageSubtitle(self, text):
        return streamlit.markdown('''
            ### '''+text+'''
        ''')

    # Removes annoying default Streamlit Top-Left Side Menu Button (Required for Online Deployment)
    def eliminateSideMenu(self):
        removeOriginalStyling = """<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"""
        streamlit.markdown(removeOriginalStyling, unsafe_allow_html=True)
        return

    # Returns Result of Sentiment Analysis from Model
    def model_Result(self, text):
        streamlit.write(" ")
        return streamlit.subheader("MODEL RESULT: "+text)

    # Returns Page Footer
    def pageFooter(self,text):
        return streamlit.markdown('''
        #### '''+text+'''
        ''')

# # load music vocab
# load_vocab_test = io.open(os.path.join(os.getcwd(), 'final_music', 'music_vocab.txt')).read()
# char_vector, vector_char, vocab_length = vectorize_music_vocab(load_vocab_test)
# # load music model and create custom model for inference
# music_model_load = music_model_arch(vocab_length, 256, 1)
# music_model_load.load_weights(os.path.join(os.getcwd(), 'final_music', 'weights'))
# music_model_load.build(tf.TensorShape([1, None]))

def load_sentiment_prediction(input_text):
    # load sentiment model
    # text needs to be preprocessed to remove break tags and strip punctuation to process further
    # this function needs to run when loading model
    @tf.keras.utils.register_keras_serializable()
    def sentiment_preprocess(data):
        data = tf.strings.lower(data)
        data = tf.strings.regex_replace(data, '<br />', ' ')
        data = tf.strings.regex_replace(data,'[%s]' % re.escape(r'!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\''), '')
        return data

    sentiment_model_load = tf.keras.models.load_model(os.path.join(os.getcwd(), 'final_sentiment'))
    prediction = sentiment_model_load.predict(np.array([input_text]))
    return prediction


# def load_music(input_text):
#     gen_text = generate_music_notation(music_model_load, char_vector, vector_char, input_text, generation_length = 2000)
#     snippet = re.findall('(^|\n\n)(.*?)\n\n', gen_text, flags=re.DOTALL)
#     gen_songs = [song[1] for song in snippet]
#     return gen_songs
#     for song_number, song in enumerate(gen_songs):
#         music_file_path = os.path.join(os.getcwd(), 'final_music', str(song_number)+input_text)
#         gen_song = convert_to_song(song, music_file_path)
#     if gen_song:
#         return gen_song


####################### RUN THE PAGE #######################
title = "Evaluate the RNN with LSTM Model on Sentiment Analysis"
prompt_sentiment="Type sequence of words (movie review or in general) to see if its positive or negative in sentiment: "

# Configure the Page Setup
streamlit.set_page_config(page_title=title, layout='wide')
# Configure Display Class
visualize=dashboard()
# Remove annoying default Streamlit Top-Left Side Menu Button (Required for Online Deployment)
visualize.eliminateSideMenu()
# Add page title
visualize.pageTitle(title)
# First Block of Language Translation Input
visualize.pageSubtitle("Determine if the Input is Postive or Negative")
# Setup user input bar
textToFeed1 = visualize.getInputText(prompt_sentiment)
# Check if user entered text input
if streamlit.button("Find Sentiment"):
    with streamlit.spinner('Wait for it...'):
        time.sleep(5)
    s = load_sentiment_prediction(textToFeed1)
    if s >=0:
        visualize.model_Result("Positive")
    else:
        visualize.model_Result("Negative")

# Page Footer
visualize.pageFooter(" ")