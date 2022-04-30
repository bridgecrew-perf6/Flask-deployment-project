import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Import the dataset
startups  = pd.read_csv("50_Startups.csv")
df = startups.copy()
df_State = pd.get_dummies(df["State"])
df_State.columns = ['California','Florida','New York']
df.drop(["State"], axis=1 , inplace =True)
df=pd.concat([df,df_State],axis=1)
df.drop(["California"], axis=1, inplace = True)
X = df.drop("Profit", axis = 1)
y = df["Profit"] 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred=lm.predict(X_test) 
import joblib
joblib.dump(lm, "multiple_regression_model.pkl")
import joblib
NewYork = 1
#California = 0
Florida = 0
RnD_Spend = 160349
Administration_Spend = 134321
Marketing_Spend = 401400
pred_args = [NewYork,Florida,RnD_Spend,Administration_Spend,Marketing_Spend]
pred_args_arr = np.array(pred_args)
pred_args_arr = pred_args_arr.reshape(1, -1)
mul_reg = open("multiple_regression_model.pkl","rb")
ml_model = joblib.load(mul_reg)
model_prediction = ml_model.predict(pred_args_arr)

round(float(model_prediction), 2)