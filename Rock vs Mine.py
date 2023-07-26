import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data = pd.read_csv("/content/sonar data.csv",header=None)
sonar_data.head()
sonar_data.shape
sonar_data.groupby(60).mean()
#seperating the data and label
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
#train and test data splitting
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)
#model training
model=LogisticRegression()
#training logistic regression on our data
model.fit(X_train,Y_train)
#train data  accuracy check chestham
X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)
#test data ki accuracy entha vachindo check chestham
X_test_prediction=model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)
#sonar nundi vachina data ni predict chestham
input_data=(0.0114,0.0222,0.0269,0.0384,0.1217,0.2062,0.1489,0.0929,0.1350,0.1799,0.2486,0.2973,0.3672,0.4394,0.5258,0.6755,0.7402,0.8284,0.9033,0.9584,1.0000,0.9982,0.8899,0.7493,0.6367,0.6744,0.7207,0.6821,0.5512,0.4789,0.3924,0.2533,0.1089,0.1390,0.2551,0.3301,0.2818,0.2142,0.2266,0.2142,0.2354,0.2871,0.2596,0.1925,0.1256,0.1003,0.0951,0.1210,0.0728,0.0174,0.0213,0.0269,0.0152,0.0257,0.0097,0.0041,0.0050,0.0145,0.0103,0.0025)#input data ni numpy array ga marusthunnam
input_data_as_numpy_array=np.asarray(input_data)
#numpy array ni reshape chestham enduku ante kastam ayyidi
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshape)
if(prediction[0]=='R'):
  print("That is Rock no problem to go")
else:
  print("That is Mine there is a problem")
