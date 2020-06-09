#imports 
from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

#import for the cl


#variables
classifier_filepath = os.path.join("./classifier/reviewClasiffierTree.pkl");
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()



#TODO move to another file called functions
#def normalize_document(text):
#    stop_words = set( nltk.corpus.stopwords.words('english')+ list(string.punctuation)+["...","*","''","``"])
#    text_without_html = BeautifulSoup(text).get_text()
#    words = text_without_html.split() 
#    words_without_contractions = [contractions.fix(word) for word in words]
#    #Join wordlist again to use word tokenize so words can be separated properly without losing meaning
#    text_complete = ' '.join(words_without_contractions)
#    words_nltk = nltk.word_tokenize(text_complete)
#    clean_words = [word.lower() for word in words_nltk if word.lower() not in stop_words]
#    clean_text = " ".join(clean_words)
#    return clean_text


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)
    
    predict_request = [['Review']] 

    predict_request = np.array(list(predict_request)).reshape(1,-1);
   
    #prediction = classifier.predict(predict_request)
    #output.append({'Diabetes': int(prediction[0])})
    # response = jsonify(output),200


    #return response
    


if __name__ == '__main__':
    app.run(port=8080, debug=True)

