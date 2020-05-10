# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Required Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix

# Import DL Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU

# Importing Dataset
df= pd.read_csv('Churn_Modelling.csv')

# Dependent and Independent variables
X = df.iloc[:,3:-1]
Y = df.iloc[:,-1].values

# Encoding the Data
X = pd.get_dummies(X,drop_first=True)

# Train and Test Split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,random_state=0,test_size=0.2)
    
# Feature Scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)

# Model Architecture
model = Sequential()
model.add(Dense(input_dim=11,init='he_uniform',activation='relu',output_dim=6))
model.add(Dense(init='he_uniform',activation='relu',output_dim=6))
model.add(Dense(init='he_uniform',activation='sigmoid',output_dim=1))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the Model
model.fit(xtrain,ytrain,batch_size=10,epochs=50)

# Predictions
pred_test = model.predict(xtest)
pred_test = pred_test>0.5

# Accuracy on Test Data
score = accuracy_score(ytest,pred_test)

# Hyper-Tuning parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(kernel_init,optimizer,activation,units):
    model = Sequential()
    model.add(Dense(input_dim=11,init=kernel_init,activation=activation,output_dim=units))
    model.add(Dense(init=kernel_init,activation=activation,output_dim=units))
    model.add(Dense(init=kernel_init,activation='sigmoid',units=1))
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
Classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'optimizer':['adam','rmsprop'],
              'activation':['relu','sigmoid'],
              'kernel_init':['uniform','he_uniform','glorot_uniform'],
              'units':[10,12,16,20]}
grid_search = GridSearchCV(estimator=Classifier,param_grid=parameters,scoring='accuracy',cv=5)
grid_search = grid_search.fit(xtrain,ytrain)

# Predictions
pred_tuned_ann = grid_search.predict(xtest)
pred_tuned_ann = (pred_tuned_ann>0.5)

# Accuracy
score_tuned = accuracy_score(ytest,pred_test)




















