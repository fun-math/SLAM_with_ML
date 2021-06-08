import tensorflow as tf 
import tensorflow.keras.layers as nn

inputs=tf.keras.Input(shape=(224,224,3))

x=nn.Conv2D(32,3,1,padding='same',activation='relu')(inputs)
x=nn.MaxPool2D(2,2)(x)

x=nn.Conv2D(32,3,1,padding='same',activation='relu')(x)
x=nn.MaxPool2D(2,2)(x)

x=nn.Conv2D(32,3,1,padding='same',activation='relu')(x)
x=nn.MaxPool2D(2,2)(x)

x=Flatten()(x)
out=Dense(10,activation='softmax')

model=tf.keras.Model(inputs=inputs,outputs=out)

model.summary()