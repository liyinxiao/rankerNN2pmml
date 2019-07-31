import keras
from keras import backend as K
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


model = RankNet_model(INPUT_DIM)
print(model.summary())

# data transformation
mms = MinMaxScaler()
mms.fit(np.concatenate((X1, X2), axis=0))
X1 = mms.transform(X1)
X2 = mms.transform(X2)

# train model
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit([X1, X2], Y, batch_size=10, epochs=5, verbose=1)

params = {
    'feature_names': ['Feature1', 'Feature2', 'Feature3'],
    'target_name': 'score',
    'copyright': 'Yinxiao Li',
    'description': 'Keras Ranknet model for demonstration purpose.',
}
rankerNN2pmml(estimator=model, transformer=mms, file='model.xml', **params)

# generate expected output
get_ranker_output = K.function([model.layers[0].input], [model.layers[-3].get_output_at(0)])
Ranker_output = get_ranker_output([X1])[0]

df_input = pd.DataFrame(X1, columns = ['Feature1', 'Feature2', 'Feature3'])
df_input.to_csv('Model_input.csv', index=False)
df_output = pd.DataFrame(Ranker_output, columns = ['score'])
df_output.to_csv('Model_output.csv', index=False)
