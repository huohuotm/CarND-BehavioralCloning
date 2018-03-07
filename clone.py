import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
import time


t0 = time.time()

# load the data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

print("Training set size:",X_train.shape)
input_shape = X_train.shape[1:]

## build a model 

model = Sequential()
model.add(Lambda(lambda x: x/255.,input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(3,5,5,activation='relu', border_mode='same'))
model.add(Conv2D(24,5,5,activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36,5,5,activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48,5,5,activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,3,3,activation='relu', border_mode='valid'))
model.add(Conv2D(64,3,3,activation='relu', border_mode='valid'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=10)


# save the model
model.save('model2.h5')
print("Training time: %.2f min" % ((time.time()-t0)/60))


## command line >> python drive.py model.h5



























