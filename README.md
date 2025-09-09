**Developing a Neural Network Regression Model**

**AIM**
To develop a neural network regression model for the given dataset.


**THEORY**

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

**Neural Network Model**
Include the neural network model diagram.

**DESIGN STEPS**

**STEP 1: Generate Dataset**
Create input values from 1 to 50 and add random noise to introduce variations in output values .

**STEP 2: Initialize the Neural Network Model**
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

**STEP 3: Define Loss Function and Optimizer**
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

**STEP 4: Train the Model**
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

**STEP 5: Plot the Loss Curve**
Track the loss function values across epochs to visualize convergence.

**STEP 6: Visualize the Best-Fit Line**
Plot the original dataset along with the learned linear model.

**STEP 7: Make Predictions**
Use the trained model to predict for a new input value .

## PROGRAM:
### Name: Gowtham S
### Register Number: 2305002008
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)

```
## DATASET INFORMATION:
"C:\Users\admin\Pictures\Screenshots\Screenshot 2025-09-09 090707.png"


## OUTPUT:
### Training Loss Vs Iteration Plot:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/a7b48087-1179-4781-8786-e3d160344202)
### Epoch Training:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/1247ecf7-80e4-4443-ab84-c09d0cd4d541)
### Test Data Root Mean Squared Error:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/0114d30a-8081-4205-a158-95efe5450804)
### New Sample Data Prediction:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/ea52cc7b-b09f-400c-90e8-a8170793c2ef)


## RESULT:
Thus a basic neural network regression model for the given dataset is written and executed successfully.
