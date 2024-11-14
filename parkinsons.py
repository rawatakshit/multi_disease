import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle


parkinsons_data = pd.read_csv('Projects/parkinsons.csv')
# print(parkinsons_data.shape)

X=parkinsons_data.drop(columns=['name','status'],axis=1)
Y=parkinsons_data['status']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

model=svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)

# print('Accuracy score of training data:',training_data_accuracy)

# X_test_prediction=model.predict(X_test)
# test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
# print('Accuracy score of test data:',test_data_accuracy)



filename='parkinsons_model.sav'
pickle.dump(model,open(filename,'wb'))