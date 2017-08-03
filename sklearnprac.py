'''
author: Sandeep Anand
- Using sklearn to used some of the libraries present
'''

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
listtrain = ['ham','spam','spam','ham','ham','spam']
listtest = ['ham','ham','spam','spam']
le.fit(listtrain)
le.classes_
le.transform(listtest)

ohe = preprocessing.OneHotEncoder()
label_encoded_data = le.fit_transform(listtrain)
print(label_encoded_data)
print(ohe.fit_transform(label_encoded_data.reshape(-1,1)))


# %%
import numpy as np
import pandas as pd
