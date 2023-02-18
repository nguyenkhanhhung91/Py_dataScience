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


warnings.filterwarnings('ignore')


data = '/home/nguyenkhanhhung91/Py_dataScience/AlgothTest/Classification/adult.csv'

df = pd.read_csv(data, header=None, sep=',\s' , nrows=10000) #reduce number of rows because svc runs very slow

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names


# Declare feature vector and target variable
X = df.drop(['income'], axis=1)

y = df['income']


#split to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

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

print('GNB Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#intuitive explanation:https://www.youtube.com/watch?v=O2L2Uv9pdDA

#confusion matrix for GNB
cm = metrics.confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=gnb.classes_)

disp.plot()

svc = SVC(C=0.001, random_state=1, kernel='linear', gamma='auto')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
# print('SVM prediction: ', y_predict[:5, :])
print('SVM Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predict)))

logr = linear_model.LogisticRegression()
logr.fit(X_train, y_train)
predicted = logr.predict(X_test)

df["GNB Prediction"] = y_pred
df["SVM Prediction"] = y_predict
df["LogisticRegression Prediction"] = predicted
df["Test label"] = y_test

print('LogisticRegression Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, predicted)))


## save to xlsx file
filepath = '/home/nguyenkhanhhung91/Py_dataScience/AlgothTest/Classification/prediction.xlsx'

df.to_excel(filepath, index=False)


# plt.show()