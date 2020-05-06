import numpy as np
from var1 import gen_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras.callbacks import Callback

EPOCHS = 10
BATCH_SIZE = 32

def load_data():
    size = 6000
    i_test = size // 5
    i_valid = 9 * size // 25
    i_train = size

    data, labels = gen_data(size)
    encoder = LabelEncoder()
    encoder.fit(labels)

    labels = encoder.transform(labels)
    k = [(data[i], labels[i]) for i in range(0, size)]
    np.random.shuffle(k)
    data = np.array(list(map(lambda x: x[0], k)))
    labels = np.array(list(map(lambda x: x[1], k)))
    data = data.reshape(*data.shape, 1)

    test_data = data[:i_test]
    test_labels = labels[:i_test]
    valid_data = data[i_test:i_valid]
    valid_labels = labels[i_test:i_valid]
    train_data = data[i_valid:i_train]
    train_labels = labels[i_valid:i_train]

    return test_data, test_labels, valid_data, valid_labels, train_data, train_labels

##############################################################################################
def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
##############################################################################################

##############################################################################################
def create_graphics(history):
    # графики потерь
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    # графики точности
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'g')
    plt.plot(epochs, val_acc, 'y')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()
##############################################################################################

##############################################################################################
class call_back(Callback):
    def __init__(self, validation_data, prefix='model', key='val_loss', date = datetime.now()):
        self.validation_data = validation_data
        self.prefix = '{}_{}_{}_'.format(date.day, date.month,date.year)+prefix+'_'
        self.loss = {}
        self.key = key
        self.index = 0

    def on_train_begin(self, logs=None):
        loss, _ = self.model.evaluate(self.validation_data[0], self.validation_data[1])
        for i in range(1, 4):
            self.loss[self.prefix + str(i)] = loss
        for key in self.loss.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs=None):
        for i in range(1, 4):
            if logs.get(self.key) < self.loss[self.prefix + str(i)] and i > self.index:
                self.loss[self.prefix + str(i)] = logs.get(self.key)
                self.model.save(self.prefix + str(i))
                self.index += 1
                break
            elif i <= self.index:
                continue
        if (self.index == 3):
            self.index = 0
        ##############################################################################################


##############################################################################################
if __name__ == '__main__':
    test_data, test_labels, valid_data, valid_labels, train_data, train_labels = load_data()
    model = build_model()
    history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(valid_data, valid_labels), callbacks=[call_back((valid_data, valid_labels))])
    create_graphics(history)
    model.evaluate(test_data, test_labels)
