import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import adam
import numpy as np

os.chdir("C:/Users/Bhavya/Documents/PRoject/AgriWC")

# Importing the dataset
dataset = pd.read_csv('data for agriculture water prediction for modelR3.csv')
#dataset.dtypes # checking datatypes of columns
X = dataset.iloc[:45,1:9].values
y = dataset.iloc[:45,9].values

# viwing pair plot
import seaborn as sns
#sns.pairplot(dataset) 

#spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)

# scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Xpred = dataset.iloc[45:47,1:9].values
Xpred= sc.transform(Xpred)
Xfull=sc.transform(X)

#np.concatenate(X_train,X_test,axis=1)
#Building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
import sklearn.metrics as metrics

# Initialising the ANN
def reg(Xtrain,YTrain):
    Regressor= Sequential()
# Adding the input layer and 2 hidden layers
    Regressor.add(Dense(output_dim = 15, kernel_initializer='normal', activation = 'relu', input_dim = 8))
    Regressor.add(Dense(output_dim = 10, kernel_initializer='normal', activation = 'relu'))
# Adding the output layer
    Regressor.add(Dense(output_dim = 1,  activation = 'linear'))
# Compiling the ANN 
    adam1=adam(lr=0.006)
    Regressor.compile(loss = 'mean_squared_error', optimizer = adam1)
    Regressor.fit(Xtrain,YTrain,epochs=500,verbose=0)   
    return Regressor

# Fitting the ANN to the Training set
estimator = reg(X_train,y_train)

#Testing on training data  
y_predtest= estimator.predict(X_train)
mse=metrics.mean_squared_error(y_train,y_predtest)
print("MSE of training data:", mse)
#Viewing weights calcluated 
weights=estimator.get_weights() 

#Visualising Neuralnet
from ann_visualizer.visualize import ann_viz
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz'
import graphviz #import Digraph
ann_viz(estimator, view=True,  title="ANN")

#Testing on heldout data   
y_predtest1= estimator.predict(X_test)
mse1=metrics.mean_squared_error(y_test,y_predtest1)
print("MSE of heldout data:", mse1)
#
## Testing on test data
ypred= estimator.predict(Xpred)

#For ploting
y1= estimator.predict(Xfull)
mse2=metrics.mean_squared_error(y,y1)

