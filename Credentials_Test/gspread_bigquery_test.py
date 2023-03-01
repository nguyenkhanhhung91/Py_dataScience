#!/usr/bin/env python3

import pandas as pd
import gspread
from google.cloud import bigquery
import numpy as np
from Modules1 import pytest
import sys

sys.path.append('/home/nguyenkhanhhung91/Py_dataScience/Modules1')



# Construct a BigQuery client object.
client = bigquery.Client()

query = """
    SELECT * FROM `bigquery-public-data.country_codes.country_codes` 
"""
query_job = client.query(query)  # Make an API request.
df_country = query_job.to_dataframe()

# print("Query results:")
# print(df_country)
# df_country.set_index("country_name", inplace=True)
# print("\nGet 1st row values")
# print(df_country.loc[['Viet Nam', 'Algeria']])

a = np.array([1,2,3])

b = a[:2]
#print(b)

c = np.flip(b)

#print('c')

# credentials = {
#     "installed":
#     {"client_id":"596469107792-rfup1ere10m1kfqok2rmtprfohut6qel.apps.googleusercontent.com",
#     "project_id":"artful-hexagon-284007","auth_uri":"https://accounts.google.com/o/oauth2/auth",
#     "token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
#     "client_secret":"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","redirect_uris":["http://localhost"]}
# }
# authorized_user = {
#     	"refresh_token": "1//0ggbtmILXU5vxCgYIARAAGBASNgF-L9IrasuCbpDzgPoR-Onrhf-UrVUtRcwVA_ijHnvLdrwmz8bdZkYfugJh9vB6Lx3NU8MnOQ", 
#         "token_uri": "https://oauth2.googleapis.com/token", 
#         "client_id": "596469107792-rfup1ere10m1kfqok2rmtprfohut6qel.apps.googleusercontent.com", 
#         "client_secret": "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", 
#         "scopes": ["https://www.googleapis.com/auth/spreadsheets", 
#         "https://www.googleapis.com/auth/drive"], 
#         "expiry": "2022-12-06T15:32:18.846123Z"
# }
# gc, authorized_user = gspread.oauth_from_dict(credentials, authorized_user)

# sh = gc.open("premier_league_table")
# ws = sh.sheet1
# df = pd.DataFrame(ws.get_all_records())

# new_column = ['Brazil','Senegal','Brazil','England','Poland','France','Spain','Spain','Denmark','Spain','Spain','Argentina','Ireland',
# 'NA','England','France','NA','England','NA', ' Nertherland']

# for i in range(len(new_column)):
# 	if new_column[i] == "NA":
# 		new_column[i] = "Unknown"

# print(new_column)

# df['country_name'] = new_column

# left_join = pd.merge(df,df_country,on = 'country_name', how = 'left')

# pd.set_option('display.width',1200)

# del left_join[df.columns[0]]

# with pd.ExcelWriter('SampleData.xlsx') as writer:
# 	left_join.to_excel(writer, sheet_name='newtab', index= 'false')