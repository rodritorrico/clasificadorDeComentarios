#imports 
from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

from cleaner import *

vectorizer_filepath = os.path.join("./Vectorizer/countVectorizer.pkl")
vectorizer_file = open(vectorizer_filepath , "rb")
vectorizer = pickle.load(open(vectorizer_filepath , "rb"))
vectorizer_file.close()


classifier_filepath = os.path.join("./classifier/reviewClasiffierTree.pkl")
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)
    predict_request = data['Review'] 
    clean_text = [normalize_document(predict_request)]
    vector = vectorizer.transform(clean_text).toarray()
    prediction = classifier.predict(vector)
    if prediction[0] == 0:
        output = "neg"
    else:
        output="pos"
    response = jsonify(output),200

    return response
    


if __name__ == '__main__':
    app.run(port=8080, debug=True)

