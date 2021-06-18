# Written by Kanchi Tank

from flask_bootstrap import Bootstrap
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd

# Import Packages
import joblib
from sklearn.feature_extraction.text import CountVectorizer

gender_app = Flask(__name__)
Bootstrap(gender_app)

@gender_app.route('/')
def index():
	return render_template('index.html')

@gender_app.route('/predict', methods=['POST'])
def predict():
	df = pd.read_csv("data/Names_dataset.csv")
	# Features and Labels
	x_df = df.name
	y_df = df.gender
    
    # Vectorization
	corpus = x_df.values.astype('U')
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) 
	
	# Loading the ML Model
	nb = open("models/naivebayes.pkl","rb")
	clf_1 = joblib.load(nb)

	# Receive the input
	if request.method == 'POST':
		name_query = request.form['name_query']
		data = [name_query]
		vct = cv.transform(data).toarray()
		my_prediction = clf_1.predict(vct)
	return render_template('results.html', prediction = my_prediction, name = name_query.upper())

if __name__ == '__main__':
	gender_app.run(debug=True)