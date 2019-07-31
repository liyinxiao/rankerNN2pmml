rankerNN2pmml
==========

Python library for converting pairwise Learning-To-Rank Neural Network models (RankNet NN, LambdaRank NN) into pmml.

## Supported model structure

It supports pairwise Learning-To-Rank (LTR) algorithms such as Ranknet and LambdaRank, where the underlying model (hidden layers) is a neural network (NN) model. 
<img src="https://github.com/liyinxiao/rankerNN2pmml/blob/master/assets/rankerNN2pmml_model.png" width=750>

## Installation
```
pip install rankerNN2pmml
```

## Example

Example on a RankNet model, with model structure as below. 

<img src="https://github.com/liyinxiao/rankerNN2pmml/blob/master/assets/RankNet_Example.png" width=750>

```python
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rankerNN2pmml import rankerNN2pmml

# generate dummy data.
INPUT_DIM = 3
X1 = 2 * np.random.uniform(size=(50, INPUT_DIM))
X2 = np.random.uniform(size=(50, INPUT_DIM))
Y = [random.randint(0,1) for _ in range(50)]

# data transformation
mms = MinMaxScaler()
mms.fit(np.concatenate((X1, X2), axis=0))
X1 = mms.transform(X1)
X2 = mms.transform(X2)

def RankNet_model(input_shape):
    # Neural network structure
    h1 = Dense(4, activation="relu", name='Relu_layer1')
    h2 = Dense(2, activation='relu', name='Relu_layer2')
    h3 = Dense(1, activation='linear', name='Identity_layer')
    # document 1 score
    input1 = Input(shape=(input_shape,), name='Input_layer1')
    x1 = h1(input1)
    x1 = h2(x1)
    x1 = h3(x1)
    # document 2 score
    input2 = Input(shape=(input_shape,), name='Input_layer2')
    x2 = h1(input2)
    x2 = h2(x2)
    x2 = h3(x2)
    # Subtract layer
    subtracted = Subtract(name='Subtract_layer')([x1, x2])
    # sigmoid
    out = Activation('sigmoid', name='Activation_layer')(subtracted)
    # build model
    model = Model(inputs=[input1, input2], outputs=out)
    return model

# build model
model = RankNet_model(INPUT_DIM)
model.compile(optimizer="adam", loss="binary_crossentropy")
# train model
model.fit([X1, X2], Y, batch_size=10, epochs=5, verbose=1)

params = {
    'feature_names': ['Feature1', 'Feature2', 'Feature3'],
    'target_name': 'score'
}
rankerNN2pmml(estimator=model, transformer=mms, file='model.pmml', **params)
```

## Params explained
* **estimator**: Keras model to be exported as PMML (see supported model structure above).
* **transformer**: if provided then scaling is applied to data fields.
* **file**: name of the file where PMML will be exported.
* **feature_names**: when provided and have same shape as input layer, features will have custom names, otherwise generic names (x<sub>0</sub>,..., x<sub>n-1</sub>) will be used.
* **target_name**: when provided target variable will have custom name, otherwise generic name **score** will be used.

## What is supported?
* Models (estimators)
    * keras.models.Model (see supported model structure above)
* Activation functions
    * tanh
    * logistic (sigmoid)
    * identity
    * rectifier (Relu)
* Transformers
    * sklearn.preprocessing.StandardScaler
    * sklearn.preprocessing.MinMaxScaler


