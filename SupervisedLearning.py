import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import BaggingClassifier

# %matplotlib inline
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
          
flatUI = ["#2c2c54", "#34ace0", "#84817a", "#ff3f34", "#05c46b", "#ffa801"]
sns.set_palette(flatUI)
sns.palplot(sns.color_palette())
plt.show()

def showFigure(fig, x=16, y=4):
    fig = plt.gcf()
    fig.set_size_inches(x, y)
    plt.show()

import numpy as np
import pandas as pd  # data processing, CSV file I/O



survey = pd.read_csv('Dataset/cleaned.csv')

y = survey['Sought Treatment']

def bestFill(datset):
    for feature in survey:
        if survey[feature].dtype == np.int64:
             print('int64, not available = -1 : ', feature)
             survey[feature].fillna(-1, inplace=True)
             survey[feature] = pd.to_numeric(survey[feature], errors='coerce').astype(int)    
        elif survey[feature].dtype == np.float64:
             print('float64, not available = -1 : ', feature)
             survey[feature].fillna(-1, inplace=True)
             survey[feature] = pd.to_numeric(survey[feature], errors='coerce').astype(float)
        elif survey[feature].dtype == np.object:
             print('object, not available = NaN : ', feature)
             survey[feature].fillna('NaN', inplace=True)
            
bestFill(survey)        

features= ['fhml',
        'cz', 
        'year',
        'age', 
        'ag', 
        'sex', 
        'pa',
        'rrp', 
        'nc',
        'ati', 
        'insurance',
        'diagnosis',
        'dmhp', 
        'remp', 
        'disorder',
        'ptemp']   

X = survey[features]

numerical_features = (X.dtypes == 'float') | (X.dtypes == 'int')
categorical_features = ~numerical_features

print(categorical_features)

print(X)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(X)
#print(df.head())


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.40, random_state=1)


# Train and evaluate model
def fit_eval_model(model, train_features, y_train, test_features, y_test):
    results = {}
  
    # Train the model
    model.fit(train_features, y_train)
   
    # Test the model
    train_predicted = model.predict(train_features)
    test_predicted = model.predict(test_features)
    
     # Classification report and Confusion Matrix
    results['classification_report'] = classification_report(y_test, test_predicted)
    results['confusion_matrix'] = confusion_matrix(y_test, test_predicted)
        
    return results
    
# Initialize the models
sv = SVC(random_state = 1)
rf = RandomForestClassifier(random_state = 1)
ab = AdaBoostClassifier(random_state = 1)
gb = GradientBoostingClassifier(random_state = 1)

gb.fit(X_train, Y_train)
result1 = gb.predict(X_test)

# Fit and evaluate models
results = {}
for cls in [sv, rf, ab, gb]:
    cls_name = cls.__class__.__name__
    results[cls_name] = {}
    results[cls_name] = fit_eval_model(cls, X_train, Y_train, X_test, Y_test)

# Print classifiers results
for result in results:
    print (result)
    print()
    for i in results[result]:
        print (i, ':')
        print(results[result][i])
        print()
    print ('-----')
    print()
# get importance
importance = gb.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (df.columns[i], v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Save the model as serialized object pickle
with open('mentaldisorderpredictionmodel.pkl', 'wb') as file:
    pickle.dump(gb, file)
########################################################################
# Validation

"""
features=[0.0, 5.0, 4.0, 44.0, 3.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0]
array_features = [np.array(features)]
print(array_features)
prediction = gb.predict(array_features)
output = prediction
print("Final OutPut")
print(output)
"""


