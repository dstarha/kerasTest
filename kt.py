import matplotlib.pyplot as plt
import time
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, rmsprop, adam
from keras.utils import np_utils
from keras import backend as K
from sklearn import preprocessing
K.set_image_dim_ordering('th')

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


MODEL_NAME = 'KerasTestModel_{}'.format(time.strftime("%d_%m_%Y_%H-%M-%S"))
TRAIN_DIR = 'yalefaces\\JPG\\'

img_rows = 128
img_cols = 128
num_channel = 1


def image_to_feature_vector(image, size=(img_rows, img_cols)):
    return cv2.resize(image, size).flatten()

img_data_list = []

img_list = os.listdir(TRAIN_DIR)
print('loading images ...')
for img in img_list:
    input_img = cv2.imread(TRAIN_DIR + '/' + img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
    #img_data_list.append(input_img_resize)
    input_img_flatten = image_to_feature_vector(input_img, (img_rows, img_cols))
    img_data_list.append(input_img_flatten)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
#img_data /= 255
print(img_data.shape)

img_data_scaled = preprocessing.scale(img_data)
print(img_data_scaled.shape)

print(np.mean(img_data_scaled))
print(np.std(img_data_scaled))

print(img_data_scaled.mean(axis=0))
print(img_data_scaled.std(axis=0))

# for theano framework
if K.image_dim_ordering() == 'th':
    img_data_scaled = img_data_scaled.reshape(img_data.shape[0], num_channel, img_rows, img_cols)
    print(img_data_scaled.shape)
# for tensorflow framework
else:
    img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
    print(img_data_scaled.shape)
# else:
#     if K.image_dim_ordering() == 'th':
#         img_data = np.rollaxis(img_data, 3, 1)
#         print(img_data.shape)

num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:999] = 0
labels[1000:1999] = 1

names = ['cats', 'dogs']

Y = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# number of samples
input_shape = img_data[0].shape

model = Sequential(name=MODEL_NAME)
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

num_epoch = 20
# same things
#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20, verbose=1, validation_data=(X_test, y_test))
hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=num_epoch, verbose=1, validation_split=0.2)

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(7,5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7,5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])


# # Generate dummy data
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))
#
# print(type(data))
# print(type(labels))
#

# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32)

# model.save(filepath=MODEL_NAME + '.h5')


