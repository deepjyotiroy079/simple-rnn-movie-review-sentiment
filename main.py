import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb



gpus = tf.config.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only use the first GPU
	try:
		tf.config.set_visible_devices(gpus[0], 'GPU')
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
	except RuntimeError as e:
		# Visible devices must be set before GPUs have been initialized
		print(e)

# Load the word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

model = load_model('movie_review_rnn.h5')

# print(model.summary())

def decode_review(text):
    return ' '.join(
        reverse_word_index.get(i - 3, '?') for i in text
    )

def preprocess_review(text):
    text = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in text]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

# creating the prediction function
def predict_sentiment(review):
	preprocessed_input = preprocess_review(review)
	prediction = model.predict(preprocessed_input)[0][0]
	sentiment = 'positive' if prediction > 0.5 else 'negative'
	return sentiment, prediction[0][0]


# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment:")
user_input = st.text_area("Review Text", "This movie was fantastic! I loved it.")

if st.button("Classify"):
	preprocess_input = preprocess_review(user_input)

	# make prediction
	prediction = model.predict(preprocess_input)
	sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
	score = prediction[0][0]

	st.write(f"Sentiment: {sentiment}")
	st.write(f"Prediction Score: {score:.4f}")

else:
	st.write("please enter a review and click 'Classify' to see the sentiment.")