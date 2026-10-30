import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_excel('C:/Users/geeta/Downloads/Housing_dataset.xlsx')
print(df.head())
print(df.tail())
print(df.sample(5))
print(df.shape)
df.info()
stats = df.describe()
print("Stats : " ,stats)
print(df.describe(include='all'))
# missing values in each column
print("Missing value:",df.isnull().sum().sum())
print("duplicated value :",df.duplicated().sum())
df = pd.get_dummies(df,drop_first= True)
X= df.drop('price',axis=1)
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
#predication
y_pred = model.predict(X_test)
#evaluating R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score:{r2:.4f}")
plt.figure(figsize=[8,8])
plt.scatter(y_test, y_pred,alpha=0.6,color='blue')
plt.xlabel('actual price')
plt.ylabel('predicted price')
plt.title('Actual vs Predicted Price')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red',lw = 2)
plt.show()