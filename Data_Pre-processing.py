
# Importing Libraries.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport

# Importing Dataset.

Df=pd.read_excel('Dataset.xlsx')

# Conversion of categorical features into numerical ones.

le=LabelEncoder()
gender=le.fit_transform(Df['Gender'])
Class=le.fit_transform(Df['CLASS'])

Df.drop('Gender',axis=1,inplace=True)
Df.drop('CLASS',axis=1,inplace=True)

Df['Gender']=gender
Df['CLASS']=Class

# print(Df.duplicated())

Df1=Df.drop_duplicates().reset_index()
Df1.drop('index',axis=1,inplace=True)

print(Df1.head())
print(Df1.info())

# Dataset report generation.

Profile_Report=ProfileReport(Df1)
Profile_Report.to_file('Profile_report_Dataset.html')

# Creating cleaned dataset.

Df1.to_excel('Cleaned_Dataset.xlsx')
