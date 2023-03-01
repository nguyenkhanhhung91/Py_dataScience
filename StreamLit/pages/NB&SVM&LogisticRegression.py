from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import warnings
import category_encoders as ce
import seaborn as sn
import matplotlib.pyplot as plt
import streamlit as st

st.title ("This App can classify income based on many features such as age, occupation, education, etc.")
st.header("Let's use GaussianNB, Supporting vector machine, and LogisticRegression to compare the accuracy.")


warnings.filterwarnings('ignore')


data = 'Data_ForStreamLit/adult.csv'

df = pd.read_csv(data, header=None, sep=',\s' , nrows=10000) #reduce number of rows because svc runs very slow

st.write('Quick overview of the data:')

st.dataframe(df.head())

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names


# Declare feature vector and target variable
X = df.drop(['income'], axis=1)

y = df['income']

selection = st.selectbox("Enter the percentage of testing size:",["0.1 or 10% for testing", "0.2 or 20% for testing", "0.3 or 30% for testing"])
Size = lambda x : 0.1 if (selection=="0.1 or 10% for testing") else  (0.2 if (selection=="0.2 or 20% for testing") else 0.3)


if st.button('Train and show accuracy score'):
	#split to training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Size(selection), random_state = 0)
	st.write('Training model with test size:', Size(selection))

	#divide the dataset to category and numberic
	categorical = [var for var in X_train.columns if X_train[var].dtype=='O']
	numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

	#print(categorical)
	#print(numerical)

	# impute missing categorical variables with most frequent value
	for df2 in [X_train, X_test]:
	    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
	    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
	    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)    

	#print(X_train[categorical].head())
	#print(X_train[numerical].head())


	#feature scaling: pre-processing to handle highly varying magnitudes or values or units of a dominant feature
	  
	#Example of feature scaling:  """ MIN MAX SCALER """
	  
	# min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
	  
	# # Scaled feature
	# x_after_min_max_scaler = min_max_scaler.fit_transform(x)
	  
	# print ("\nAfter min max Scaling : \n", x_after_min_max_scaler)
	  
	 #Robust Scaling:

	 
	df = DataFrame(X_test)

	scaler = RobustScaler()

	encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
	                                 'race', 'sex', 'native_country'])

	X_train = encoder.fit_transform(X_train)



	X_test = encoder.transform(X_test)

	cols = X_train.columns
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test) 
	X_train = pd.DataFrame(X_train, columns=[cols])
	X_test = pd.DataFrame(X_test, columns=[cols])



	#print(X_train.head())

	gnb = GaussianNB()

	# fit the model
	gnb.fit(X_train, y_train)

	y_pred = gnb.predict(X_test)

	st.write('GNB Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

	#intuitive explanation:https://www.youtube.com/watch?v=O2L2Uv9pdDA

	#confusion matrix for GNB
	cm = metrics.confusion_matrix(y_test,y_pred)

	disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=gnb.classes_)

	disp.plot()

	svc = SVC(C=0.001, random_state=1, kernel='linear', gamma='auto')
	svc.fit(X_train, y_train)
	y_predict = svc.predict(X_test)
	# print('SVM prediction: ', y_predict[:5, :])
	st.write('SVM Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predict)))

	logr = linear_model.LogisticRegression()
	logr.fit(X_train, y_train)
	predicted = logr.predict(X_test)

	df["GNB Prediction"] = y_pred
	df["SVM Prediction"] = y_predict
	df["LogisticRegression Prediction"] = predicted
	df["Test label"] = y_test

	st.write('LogisticRegression Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, predicted)))

	st.write('The following table include the test label and prediction from all 3 algorithms')

	st.dataframe(df)



st.write('''
The math behind NB is based on fact that when both event A and B happens and they are dependent on each other then:
The probability of B happens + (probability of A happen given B already happen) =  the probability of A happens + (probability of B given A)
Move "The probability of B happens" to the left side and you have the NB formula. 
If you have tranning data set then you can figure out the frequency of event 
eg: 5 out 7 email given title include "money" is spam. the p of spam given "money" is 5/7. 
If there are 50 spam emails out of 100 total emails, then p of spam is 0.5.
Note that NB uses bag of word model (no order).
	''')

st.write('''
The objective of a SVM is to find the optimal separating hyperplane. Let say we pick a hyperplane (or line in 2D), then double it. 
We will have another line called margin. If the line/hyperplane is far from the closest data point, then margin will be big
If the line/hyperplane is close from the closest data point, then margin will be small. 
We want to maximize the margin so instead of double we can increase the margin. There can also be soft margin where some data is inside the margin.
	''')


st.write('''
Logistic Regression is used when our dependent variable is dichotomous or binary. It just means a variable that has only 2 outputs.
(NB and SVM are similar but they can be modified for multiclass classification though)
We use the logistic function to convert the output to somewhere between 0 and 1. 
If the output from logistic function is greater than the threshold say 0.5 then we predict 1, if it is smaller than the threshold then we predict 0 for y.
	''')