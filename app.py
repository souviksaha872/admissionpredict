from numpy import result_type
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lr_model.pickle', 'rb'))
@app.route('/')
def home():
	return render_template('admission.html')
@app.route('/predict', methods=['GET','post'])
def predict():
	
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	
	CGPA = float(request.form['CGPA'])
	
	
	final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, CGPA]])
	
	predict = model.predict(final_features)
	
	
	return render_template('admission.html', prediction_text='Admission chances are {}'.format(f"{predict[0] * 100:.2f}%"))
	
if __name__== "__main__":
	app.run(debug=True)