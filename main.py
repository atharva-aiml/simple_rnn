import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the imdb dataset word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation

model = load_model('Simple_rnn_imdb.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 2: Helper Functions
# Function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to process user input

def preprocess_text(text, maxlen=500, vocab_size=10000):
    words = text.lower().split()
    # Map words -> index (use 2 if not in vocab)
    encoded_review = [word_index.get(w, 2) for w in words if word_index.get(w, 2) < vocab_size]
    # Shift indices by +3 only ONCE (IMDB convention)
    encoded_review = [i + 3 for i in encoded_review]
    # Add "start" token = 1
    encoded_review = [1] + encoded_review
    # Pad/truncate
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review


#Step 3: Creating prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]> 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

    #Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Plese enter a movie review')
