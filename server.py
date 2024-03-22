# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api/',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    
    # Make prediction using model loaded from disk as per the data.

    predict_request=[[data['Pclass'],data['Sex'],data['SibSp'],data['Parch']]]
    request1=np.array(predict_request)
    print(request1)
       
    prediction = model.predict(predict_request)

    # Take the first value of prediction
    output = prediction[0]
    output = int(output)
    result_dict ={"prediction":output}
    return result_dict


if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True, host="0.0.0.0")
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
