# Program written by MT19AIE323 #


import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd  # data processing, CSV file I/O
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.axes._axes import _log as matplotlib_axes_logger

# Load ML model
model = pickle.load(open('mentaldisorderpredictionmodel.pkl', 'rb')) 

#model = pickle.load(open('mentaldisorderpredictionmodel_pkl.pkl', 'rb')) 
                
# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('MentalDisorderCheck.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    print("features")
    print(features)
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('MentalDisorderCheck.html', 
                               result = 'The patient is likely to have mental disorder!')
    else:
        return render_template('MentalDisorderCheck.html', 
                               result = 'The patient is not likely to have mental disroder!')

if __name__ == '__main__':
#Run the application
    app.run()
    
    
