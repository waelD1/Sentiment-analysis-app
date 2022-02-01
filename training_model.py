import keras
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction import text
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
from tensorflow.keras.callbacks import EarlyStopping
import pickle

#max len of the text
max_len_text = 250
trunc_type = 'post'

# load the dataset
data = pd.read_csv('Reviews.csv', nrows = 100000)

# Drop duplicates and NA
data.drop_duplicates(subset = ['Text'], inplace = True)
data.dropna(axis=0,inplace=True)

# Only taking very bad reviews (1 star or 2 stars) or very good reviews (4 stars or 5 stars)
data = data[data['Score'] != 3]

# Creating the label with 1 if it is a good review, 0 otherwise
data['review_score'] = np.where((data['Score'] ==4) |(data['Score'] ==5), 1, 0)

# Create the function to remove punctuation
def remove_punct(text):
    text =  "".join([elem for elem in text if elem not in string.punctuation]).lower()
    return text

# Application of the function
data['Text_new'] = data['Text'].apply(lambda x : remove_punct(x))

# Splitting training, validation and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data['Text'], data['review_score'], test_size=0.2, random_state = 42, shuffle = True)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

# Create a tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Creation of the index
word_index = tokenizer.word_index

### Training part

# Encoding part
X_train_tok = tokenizer.texts_to_sequences(X_train)
X_val_tok = tokenizer.texts_to_sequences(X_val)
X_test_tok = tokenizer.texts_to_sequences(X_test)

# Padding part
X_train_pad = sequence.pad_sequences(X_train_tok, maxlen=max_len_text, truncating=trunc_type)
X_val_pad = sequence.pad_sequences(X_val_tok, maxlen=max_len_text, truncating=trunc_type)
X_test_pad = sequence.pad_sequences(X_test_tok, maxlen=max_len_text, truncating=trunc_type)

# Defining vocabulary size
voc_size = len(list(word_index)) + 1

# Creation of the model
model = keras.Sequential()
model.add(keras.layers.Embedding(voc_size, 16))
model.add(keras.layers.GlobalAveragePooling1D()) # Reduce the dimension
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Prints a summary of the model
model.summary()  

# Compilation of the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Definition of the early stopage of the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

# Fit the model with the data
history = model.fit(X_train_pad, Y_train, epochs=10, batch_size=520, validation_data=(X_val_pad, Y_val), callbacks=[es], verbose=1)

# Evaluate the model
results = model.evaluate(X_test_pad, Y_test)
print(results)

# Save the model
#model.save('new_model.h5')

# Save the word index to use it to make prediction
#with open('word_index.pkl', 'wb') as f:
#    pickle.dump(word_index, f)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### Example of predictions ### ### ### ### ### ### ### ### ### ### ### ### ### ### 


# Creation of a encoding function
def encode_text(text):
  text = remove_punct(text)
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 1 for word in tokens]
  return sequence.pad_sequences([tokens], max_len_text)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

# Creation of a decoding function
reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))

# Now we can make some predictions
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  print(result[0])

#positive_review = "The food was excellent. I highly recommend this restaurant, one of the best in the area"
#predict(positive_review)

#negative_review = "The food was disgusting. I will never go to this restaurant again"
#predict(negative_review)
