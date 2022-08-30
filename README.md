# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains nine neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model. Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data.We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate.Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.After fitting model, we can test it on test data to check whether the case of overfitting.

## Neural Network Model

![image](https://user-images.githubusercontent.com/112486797/187452781-b524d476-632f-42d1-8bf3-f1a649e832ef.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
Developed by: K.Garshan kumar
Registration number: 212219220027
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv("Book1.csv")
data.head()
x=data[['input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
MinMaxScaler()
Scaler.fit(x_test)
MinMaxScaler()
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI=Sequential([
Dense(9,activation='relu'),
Dense(7,activation='relu'),
Dense(1,activation='relu')
])
AI.compile(optimizer='rmsprop',loss='mse')
loss_df=pd.DataFrame(AI.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
x_test1=Scaler.transform(x_test)
x_test1
x_test1=Scaler.transform(x_test)
x_test1
x_test1=Scaler.transform(x_test)
x_test1
AI.evaluate(x_test1,y_test)
x_n1=[[29]]
x_n1_1=Scaler.transform(x_n1)
AI.predict(x_n1_1)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/112486797/187454584-bb4c2216-3996-4b6e-81dd-a0061e1cbc25.png)


## OUTPUT

![image](https://user-images.githubusercontent.com/112486797/187455160-4a97ead4-d62f-41d6-a7c0-23bd79643bc9.png)




### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/112486797/187455478-0e36bc4c-eb52-4aab-9770-2397ddcf37cd.png)


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/112486797/187455614-76dcc90e-9012-49f9-9f04-95738a5d3911.png)




## RESULT
Thus,the model has been developed successfully.
