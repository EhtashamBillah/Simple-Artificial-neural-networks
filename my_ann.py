import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix



# Loading the data
df = pd.read_csv("Churn_Modelling.csv")
df.head()
df.shape
df.dtypes
df.describe()
df.corr()
df.skew()

# Data preprocessing
# Dummy varibale
cat_features = ["Geography","Gender"]
df_final = pd.get_dummies(data = df,columns = cat_features,drop_first = True)

x = df_final.iloc[:,3:].drop("Exited",axis = 1)
y = df_final["Exited"]


# Splitting the data
test_size = 0.20
seed = 2019
x_train,x_test,y_train,y_test = train_test_split(scaled_x, y, 
                                                 test_size = test_size,
                                                 random_state = seed)

# scalling
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

scaled_x_train = pd.DataFrame(data = scaled_x_train,columns=x_train.columns)
scaled_x_test = pd.DataFrame(data = scaled_x_test,columns=x_test.columns)

# Correlation plot
sns.heatmap(scaled_x_train.corr())

# Building ann
# Sequential module to initilize the neural network
# Dense module to build the layers of ann
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing ann
ann_classifier = Sequential()

# adding the input layer and firts hiden layer.
# rule of thumb: number of nodes in hidden layer = average number of nodes 
# in the input layer and number of nodes in output layer. In this case, we have 
# 11 input node and 1 output node. so number of hidden node = (11+1)/2 = 6. If we
# dont want to use rule of thumb, then we can experiment with different number
# of nodes using cross validation nad parameter tuning.

# The following network contains a input layer(11 nodes) and a hidden layer(6 nodes)
ann_classifier.add(Dense(input_dim = 11,                      # no. of nodes in input layer = number of independent variables
                         units = 6,                           # number of nodes in 1st hiden layer
                         kernel_initializer= "uniform",       # initializing the weights with uniform distribution
                         activation= "relu"                   # activation function in hidden layer        
                         ))

# Now adding the 2nd hidden layer
ann_classifier.add(Dense(units = 6, # here input_dim is not required beacuse 2nd hidden layer is conneted to 1st hidden layer, not to input layer
                         kernel_initializer= "uniform",
                         activation= "relu"
                         ))

# Finally adding the output layer
ann_classifier.add(Dense(units = 1, # no of units in output layer (p=yes/poitive/1)
                         kernel_initializer= "uniform",
                         activation= "sigmoid"
                         ))

# Now compiling the whole artificial neural network 
#(Here we'll apply stochastic gradient descent on the whole network)
ann_classifier.compile(optimizer = "adam", #adam is one of sgd method
                       loss = "binary_crossentropy", # loss function we want to minimize for 2-class classification
                       metrics = ["accuracy"]
                       )

# Fitting th ann model
ann_classifier.fit(scaled_x_train,
                   y_train,
                   batch_size = 10, # no of obs. after which we want to update the weights
                   epochs = 100 
                   )

# Prediction
yhat_prob = ann_classifier.predict(scaled_x_test)
yhat_pred = yhat_prob> 0.5

# Confusion metrix
print(confusion_matrix(y_test,yhat_pred))
print(classification_report(y_test,yhat_pred))

