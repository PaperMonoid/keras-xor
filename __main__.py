from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# create model with 2 inputs and 1 output
model = Sequential()

model.add(Dense(2, input_shape=(2,)))
model.add(Activation("sigmoid"))

model.add(Dense(3))
model.add(Activation("sigmoid"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

# use loss function for binary classification
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# training data
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])

# train model
model.fit(data, labels, epochs=5000, batch_size=4)

# test data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# predict classes
classes = model.predict(inputs)

# print results
print("INPUTS")
print(inputs)

print("OUTPUTS")
print(classes)
