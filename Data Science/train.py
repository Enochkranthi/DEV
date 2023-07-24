# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/Salary_Data.csv', inferSchema=True, header=True)

df = df.toPandas()

# COMMAND ----------

df

# COMMAND ----------

df = df.dropna()

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# COMMAND ----------

X = df[['YearsExperience','Age']]
y = df['Salary']

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# COMMAND ----------

lr = LinearRegression()
lr.fit(X_train,y_train)

# COMMAND ----------

y_predict = lr.predict(X_test)
r2_score(y_test, y_predict)

# COMMAND ----------

import pickle

# COMMAND ----------

filename = '../Model/model.pkl'
pickle.dump(lr, open(filename, 'wb'))

# COMMAND ----------

pip freeze > requirements.txt
