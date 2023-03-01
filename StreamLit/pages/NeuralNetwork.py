 
import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st

st.header('Running Neural Network on Google Colab can avoid installing keras and tensorflow on my desktop.')
st.write('Please click on the links belows to go directly to Google Colab:')


decisiontree = "https://colab.research.google.com/drive/1_OrcCHyFiG8HzDs0PPJpIzsMR9bPNSeM?usp=share_link"
st.write("check out this [link](%s) for Keras Decision Tree model" % decisiontree)


lstm = "https://colab.research.google.com/drive/1hPaPhgrgCDKUytAHeC0Ogpaa4Stk8YOw?usp=sharing"
st.write("check out this [link](%s) for KerasLSTM model" % lstm)


neuralnetwork = "https://colab.research.google.com/drive/1Fui1TKy7kkj5KrF2QL1xI0hNZEkGlWsM?usp=sharing"
st.write("check out this [link](%s) for Keras simple neural network " % neuralnetwork)


pyspark = "https://colab.research.google.com/drive/12rOv0HUFI_mNN2saSyG8z-_QBGdNkUyu"
st.write("check out this [link](%s) for pyspark" % pyspark)


# st.write('''Suppose this is the survey data from 500 people:

# 3 Columns represent Republican Democrat Independent	
# 2 Rows represent Male and Female
# ''') 

# data = [[120, 90, 40],
#         [110, 95, 45]]
# st.write(np.array(data))

# st.write('Using stats.chi2_contingency algorithm, the result we get is ', stats.chi2_contingency(data))

# st.write('''The way to interpret the output is as follows:

# Chi-Square Test Statistic: 0.864
# p-value: 0.649
# Degrees of freedom: 2 (calculated as #rows-1 * #columns-1)
# Array: The last array displays the expected values for each cell in the contingency table.
# Recall that the Chi-Square Test of Independence uses the following null and alternative hypotheses:

# H0: (null hypothesis) The two variables are independent.
# H1: (alternative hypothesis) The two variables are not independent.
# Since the p-value (.649) of the test is not less than 0.05, we fail to reject the null hypothesis.
# This means we do not have sufficient evidence to say that there is an association between gender and political party preference.
# ''') 