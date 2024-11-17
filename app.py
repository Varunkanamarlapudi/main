from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from gensim.models import Word2Vec
import pickle
import numpy as np
import string
from tensorflow.keras.utils import register_keras_serializable

app = Flask(__name__)

@register_keras_serializable()
class SumAlongAxis(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

try:
    model = tf.keras.models.load_model(
        'amazon_sentiment_model.keras',
        custom_objects={'SumAlongAxis': SumAlongAxis}
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile after loading
    wv_model = Word2Vec.load('word2vec_model.bin')

    with open('stop_words.pkl', 'rb') as f:
        stop_words = pickle.load(f)
except Exception as e:
    print(f"Error loading models or data: {e}")

def remove_punctuation(s):
    """Remove punctuation from a string."""
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

def remove_stop_words(raw_sen, stop_words):
    """Remove stop words from a list of words."""
    return [w for w in raw_sen if w not in stop_words]

def predict_sentiment(comment):
    comment = remove_punctuation(comment)
    comment = remove_stop_words(comment.split(), stop_words)
    word_set = set(wv_model.wv.index_to_key)
    valid_words = [w for w in comment if w in word_set]

    if not valid_words:  # No recognizable words
        return 0.4  # Neutral score for random or non-meaningful input

    X = np.zeros((1, 25, 100))
    nw = 24
    for w in list(reversed(valid_words)):
        if w in word_set and nw >= 0:
            X[0, nw] = wv_model.wv[w]
            nw -= 1

    prediction = model.predict(X)
    return float(prediction[0][0])

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')
@app.route('/demo')
def demo():
    """Render the demo page."""
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    data = request.get_json(force=True)
    comment = data['comment']
    sentiment = predict_sentiment(comment)
    label = 'Positive' if sentiment > 0.6 else 'Negative' if sentiment < 0.4 else 'Please Give Valid-Comment'
    return jsonify({'sentiment': sentiment, 'label': label})

if __name__ == '__main__':
    app.run(debug=True)
    