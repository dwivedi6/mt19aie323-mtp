
"""
Name : Sushil Kumar Dwivedi
Roll N0. MT19AIE323
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
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

def Fn_HandlingMissing(datset):
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
            
Fn_HandlingMissing(survey)        

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
        'diagnosis',
        'insurance',
        'dmhp', 
        'remp', 
        'disorder',
        'ptemp']   

X = survey[features]

numerical_features = (X.dtypes == 'float') | (X.dtypes == 'int')
categorical_features = ~numerical_features

#print(categorical_features)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
preprocess = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(), categorical_features), remainder="drop", n_jobs= -1, verbose = True
)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.40, random_state=1)

# Accuracy of each model
models = {}


from sklearn.pipeline import make_pipeline
from sklearn import metrics
def Fn_ModelEvaluation(model, yPredClass, plot=False):
    # Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(Y_test, yPredClass)
    print('Classification Accuracy:', accuracy*100)
    
    # Comparing the true and predicted response values
    print('\nTrue:', Y_test.values[0:25])
    print('Pred:', yPredClass[0:25])
   
    # Metrics computed from a confusion matrix
    confusion = metrics.confusion_matrix(Y_test, yPredClass)
    # [row, column]
    TP = confusion[1, 1]   # True Positive
    TN = confusion[0, 0]   # True Negative
    FP = confusion[0, 1]   # False Positive
    FN = confusion[1, 0]   # False Negative
    
    # False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(FP+TN)
    print('\nFalse Positive Rate:', false_positive_rate)

    # Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(Y_test, yPredClass))

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(Y_test, yPredClass))
    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # visualize Confusion Matrix

    fig = sns.heatmap(confusion, annot=True, fmt="d")
    bottom, top = fig.get_ylim()
    fig.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # histogram of predicted probabilities
    if plot:
        plt.rcParams['font.size'] = 12
        plt.hist(y_pred_prob, bins=4)
        # x-axis limit from 0 to 1
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')

    # AUC is the percentage of the ROC plot that is underneath the curve
    # Higher value = better classifier
    roc_auc = metrics.roc_auc_score(Y_test, y_pred_prob)

    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_prob)
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy

from sklearn.linear_model import LogisticRegression
def Fn_logRegression():
    # train a logistic regression model on the training set
    modelLogisticRegression = make_pipeline(preprocess, LogisticRegression(solver='liblinear', multi_class='ovr'))
    modelLogisticRegression.fit(X_train, Y_train)
    # make class predictions for the testing set
    y_pred_class = modelLogisticRegression.predict(X_test)
    print('############### Logistic Regression ###############')
    accuracy_score = Fn_ModelEvaluation(modelLogisticRegression, y_pred_class, True)
    models['Logistic Regression'] = accuracy_score * 100    
Fn_logRegression()


from sklearn.neighbors import KNeighborsClassifier
def Fn_kNN():
    modelKNN = make_pipeline(preprocess, KNeighborsClassifier())
    modelKNN.fit(X_train, Y_train)
    # make class predictions for the testing set
    y_pred_class = modelKNN.predict(X_test)
    print('########### K-Neighbors Classifier ###############')
    accuracy_score = Fn_ModelEvaluation(modelKNN, y_pred_class, True)
    # Data for final graph
    models['K-Neighbors Classifier'] = accuracy_score * 100
Fn_kNN()


from sklearn.tree import DecisionTreeClassifier
def Fn_decisionTree():
    # train a logistic regression model on the training set
    modelDecisionTreeClassifier = make_pipeline(preprocess, DecisionTreeClassifier())
    modelDecisionTreeClassifier.fit(X_train, Y_train)
    # make class predictions for the testing set
    y_pred_class = modelDecisionTreeClassifier.predict(X_test)
    print('########### Decision Tree Classifier ###############')
    accuracy_score = Fn_ModelEvaluation(modelDecisionTreeClassifier, y_pred_class, True)
    models['Decision Tree Classifier'] = accuracy_score * 100    
Fn_decisionTree()


from sklearn.svm import SVC
def Fn_SVM():
    # train a logistic regression model on the training set
    modelSVM = make_pipeline(preprocess, SVC(gamma='auto', probability=True))
    modelSVM.fit(X_train, Y_train)
    # make class predictions for the testing set
    y_pred_class = modelSVM.predict(X_test)
    print('########### C-Support Vector Classification ###############')
    accuracy_score = Fn_ModelEvaluation(modelSVM, y_pred_class, True)
    models['C-Support Vector Classification'] = accuracy_score * 100    
Fn_SVM()


from sklearn.ensemble import BaggingClassifier
def Fn_bagging():
    modelBagging = make_pipeline(preprocess, BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5))
    modelBagging.fit(X_train, Y_train)    
    # make class predictions for the testing set
    y_pred_class = modelBagging.predict(X_test)
    print('########### Bagging meta-estimator ###############')
    accuracy_score = Fn_ModelEvaluation(modelBagging, y_pred_class, True)
    models['Bagging'] = accuracy_score * 100    
Fn_bagging() 


from sklearn.ensemble import AdaBoostClassifier
def Fn_adaboost():
    modelAdaBoost = make_pipeline(preprocess, AdaBoostClassifier(n_estimators=100))
    modelAdaBoost.fit(X_train, Y_train)  
    # make class predictions for the testing set
    y_pred_class = modelAdaBoost.predict(X_test)
    print('########### Boosting meta-estimator ###############')
    accuracy_score = Fn_ModelEvaluation(modelAdaBoost, y_pred_class, True)
    models['AdaBoost'] = accuracy_score * 100
    with open('mentaldisorderpredictionmodel_pkl.pkl', 'wb') as files:
        pickle.dump(modelAdaBoost, files)
Fn_adaboost()

accuracyScores = pd.Series(models)
fig = sns.barplot(x=accuracyScores.index, y=accuracyScores.values)
plt.ylim([50.0, 85.0])
for p in fig.patches:
        fig.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.1, p.get_height() * 1.01))
showFigure(fig)

Dict_var = dict(zip(accuracyScores.index,accuracyScores.values))  
#for i,j in Dict_var.items():
#    print(i,":",j)
Keymax = max(Dict_var, key= lambda x: Dict_var[x])
print('Best Prediction Model is :', Keymax)
