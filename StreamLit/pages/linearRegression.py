#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as mticker
from matplotlib.ticker import NullFormatter
import pydotplus
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


st.title ("This App can show the relationship between interest_rate, unemployment_rate, and index_price.")
st.header ("It uses the linear regression model.")
#Let find which data file we can use for forecasting, it should not include a lot of variance and has a lot of variables. (least square)

# df0 = pd.read_json('Data_ForStreamLit/VietnamExportToChina.json')
# df0 = df0.drop('symbol', axis = 1)

# #print(df.describe())
# # print(df.to_string)

# year1 = df0['date']
# value1 = df0['value']

# df = pd.read_json('Data_ForStreamLit/ChinaExportToVietnam.json')
# df = df.drop('symbol', axis = 1)

# year2 = df['date']
# value2 = df['value']


# figure, axis = plt.subplots(1, 2, squeeze=False)


# axis[0, 0].plot(year1, value1, label="VietnamExportToChina")
# axis[0, 0].plot(year2, value2, label="ChinaExportToVietnam")
# axis[0, 0].set_title = "Vietnam And China"
# axis[0, 0].legend()

# df = pd.read_json('Data_ForStreamLit/ChinaExportToUSA.json')
# df = df.drop('symbol', axis = 1)
# year3 = df['date']
# value3 = df['value']

# df = pd.read_json('Data_ForStreamLit/VietnamExportToUSA.json')
# df = df.drop('symbol', axis = 1)
# year4 = df['date']
# value4 = df['value']

# axis[0, 1].plot(year3, value3, label="ChinaExportToUSA")
# axis[0, 1].plot(year4, value4, label="VietnamExportToUSA")
# axis[0, 1].set_title = "Vietnam And China export to USA"
# axis[0, 1].ticklabel_format(axis="y", style='plain')
# axis[0, 1].legend()

# # plt.show()


# #From the plot shown above, it seems like Vietnam export to China is pretty steady and linear after 2014, not seasonal and influcuate even during Covid time
# #So we use this df to test the forecasting algo

# train = df0[(df0.date < pd.to_datetime("2023-11-01", format='%Y-%m-%d')) & (df0.date > pd.to_datetime("2014-11-01", format='%Y-%m-%d'))]
# test = df0[df0.date > pd.to_datetime("2021-11-01", format='%Y-%m-%d')] 


# X=pd.to_datetime(train['date'], format='%Y-%m-%d')

# Y = np.array(train['value'].values, dtype=float)

# regr = linear_model.LinearRegression()
# model = regr.fit(X.values.reshape(-1, 1), Y)

# predictions = regr.predict(X.values.astype(float).reshape(-1, 1))


# axis[0, 0].plot(X, predictions,label='Linear fit', lw=3)
# axis[0, 0].scatter(X, Y,label=' real value', marker='o', color='r')
# axis[0, 0].legend()
# axis[0, 0].ticklabel_format(axis="y", style='plain')

# #Formating values in billions
# def formatter(x, pos):
#     return str(round(x / 1e9, 1))

# axis[0, 0].yaxis.set_major_formatter(formatter)
# axis[0, 0].yaxis.set_minor_formatter(NullFormatter())
# axis[0, 1].yaxis.set_major_formatter(formatter)
# axis[0, 1].yaxis.set_minor_formatter(NullFormatter())

# #predictions = float(predictions.values)
# print(Y)
# print(predictions)

# mae = mean_absolute_error(Y , predictions)
# mse = mean_squared_error(Y , predictions)
# rmse = np.sqrt(mse)

# print(f'Mean absolute error: {mae:.2f}')
# print(f'Mean squared error: {mse:.2f}')
# print(f'Root mean squared error: {rmse:.2f}')

# plt.show()


#multi variable data example
data = {'year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
        'month': [12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
        'interest_rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
        'unemployment_rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
        'index_price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
        }

df = pd.DataFrame(data)
df_selectedcols = df[['interest_rate','unemployment_rate','index_price']]
#ploting 3D
st.write('Here is our historical data and the relationship. ')
st.dataframe(df)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x = df['interest_rate']
# y = df['unemployment_rate']
# z = df['index_price']
# ax.scatter3D(x, y, z)
# ax.set_title = "How interest rate, unemployment rate affect index price?"
# ax.set_zlabel('index_price')
# ax.set_xlabel('interest_rate')
# ax.set_ylabel('unemployment_rate')
# fig = ax.figure
# st.plotly_chart(fig)


fig = px.scatter_3d(df_selectedcols, x='interest_rate', y='unemployment_rate', z='index_price')
st.plotly_chart(fig)

#Prediction model
x = df[['interest_rate','unemployment_rate']]
y = df['index_price']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)
st.write('Intercept: \n', regr.intercept_)
st.write('Coefficients: \n', regr.coef_)

st.write("Now, let's predict")

interest_rate = st.text_input('Please enter interest rate:')
unemployment_rate = st.text_input('Please enter unemployment rate:')


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

if st.button('Predict'):
	if interest_rate == "" or  unemployment_rate == "":
		st.write("The interestrate or unemployment rate is emtpy. Please enter again.")
	elif not isfloat(interest_rate):
		st.write("Oops! Number only.  Try again...")
	elif not isfloat(unemployment_rate):
		st.write("Oops! Number only.  Try again...")
	else:
		x = [interest_rate,unemployment_rate]
		prediction = regr.predict(np.array(x).reshape(-1, 2))
		st.write("The index price predicted is: ",prediction[0])

