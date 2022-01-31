from flask import Blueprint, render_template, request, flash
import tensorflow as tf
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 
from sklearn.feature_extraction import text
import string



max_len_text = 250

stop_words = text.ENGLISH_STOP_WORDS

#load the word index of the model
with open('model_summarization/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

#loading the model
model_loaded = tf.keras.models.load_model('model_summarization/sentiment_analysis_model.h5')

# Preprocessing function that remove the punctuation
def remove_punct(text):
    text =  "".join([elem for elem in text if elem not in string.punctuation]).lower()
    #text =  " ".join([word for word in text.split() if not word in stop_words])
    return text

# Create a function to automaticaly encode the text
def encode_text(text):
  text = remove_punct(text)
  tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 1 for word in tokens]
  return pad_sequences([tokens], max_len_text)[0]

# Decode function
reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]

# function to make a prediction
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model_loaded.predict(pred) 
  score = round(np.amax(result), 3)
  print('this is the score of the model : ', score)
  return score

#positive_review = "That movie was great! really loved it and would great watch it again because it was amazingly great"
#predict(positive_review)

#negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
#predict(negative_review)

