import numpy as np
import flask
from flask import Flask,request,jsonify,render_template
import pickle
import streamlit as st

st.session_state['answer'] = ''
if  st.session_state['answer'] in realans:
        answerStat = "correct"
   elif st.session_state['answer'] not in realans:
        answerStat = "incorrect"


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model.predict(final_features)
    output=prediction
    return render_template('index.html',pred_text='mpg should be{}'.format(output))

if __name__=='__main__':
    app.run(debug=True)

