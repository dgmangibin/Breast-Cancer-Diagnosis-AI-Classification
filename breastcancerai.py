import pandas as pd
import tensorflow as tf
#we are creating a neural network - has an input layer (the attributes of the cancer data set)
#goes through a hidden layer -- output layer - breast cancer diagnosis 0 = no, 1 = yes

BSdataset = pd.read_csv('/content/data.csv') #Breast Cancer Prediction
x = BSdataset.drop(columns=["diagnosis",]) 
y = BSdataset["diagnosis"]
from sklearn.model_selection import train_test_split #splitting data into a testing set and a training set to avoid overfitting

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()
#Dense are standard neurons that you get in keras 
#activation all values from neural network plotted between 0 and 1 -helps reduce the model complexity

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)
