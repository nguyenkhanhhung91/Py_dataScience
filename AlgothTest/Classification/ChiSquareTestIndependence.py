 
import pandas as pd
import numpy as np
import scipy.stats as stats

#Suppose this is the survey data from 500 people
# 			Republican	Democrat	Independent	Total
# Male		120			90			40			250
# Female	110			95			45			250
# Total		230			185			85			500

data = [[120, 90, 40],
        [110, 95, 45]]

print(stats.chi2_contingency(data))

# The way to interpret the output is as follows:

# Chi-Square Test Statistic: 0.864
# p-value: 0.649
# Degrees of freedom: 2 (calculated as #rows-1 * #columns-1)
# Array: The last array displays the expected values for each cell in the contingency table.
# Recall that the Chi-Square Test of Independence uses the following null and alternative hypotheses:

# H0: (null hypothesis) The two variables are independent.
# H1: (alternative hypothesis) The two variables are not independent.
# Since the p-value (.649) of the test is not less than 0.05, we fail to reject the null hypothesis.
# This means we do not have sufficient evidence to say that there is an association between gender and political party preference.