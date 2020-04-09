import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from PIL import Image
from tensorflow.keras import optimizers

NUM_EPOCHS = 20

optimizers = [optimizers.Adadelta(learning_rate=0.01),
              optimizers.Adagrad(learning_rate=0.01),
              optimizers.Adam(),
              optimizers.RMSprop(),
              optimizers.Ftrl(),
              optimizers.SGD(momentum=0.9)]

##############################################################################################
def load_data():
    #MNIST - набор данных
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()

    #преобразование изображений в масиив чисел из интервала [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #кодирование метк категорий
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels
##############################################################################################

##############################################################################################
# Функция для загрузки изображения не из датасета
def transform_Image(filename):
    img = Image.open(filename).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = 1 - img
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
##############################################################################################

##############################################################################################
# Задание базовой архитектуры сети
def build_model(opt):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #задание ф-ции потерь, оптимизатора, метрики
    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

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
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # графики точности
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'g')
    plt.plot(epochs, val_acc, 'y')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
##############################################################################################

##############################################################################################
def train_model():
    model = build_model('adam')
    # обучение сети
    history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, batch_size=128, validation_data=(test_images, test_labels))
    create_graphics(history)

    # проверка, как модель распознает контрольный набор
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    return model
##############################################################################################

##############################################################################################
def test (opt):
    opt_config = opt.get_config()
    print ("Researching with optimizer ")
    print(opt_config)

    model = build_model(opt)
    history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, batch_size=128, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    create_graphics(history)

    result["%s" % (opt_config)] = test_acc
    print(test_loss)
    return model
##############################################################################################

##############################################################################################
if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()

    print("Введите число:")
    print("0 - Запустить модель с наилучшими параметрами")
    print("1 - Запустить все")
    num = input()
    if num == '0':
        model = train_model()
        image = transform_Image('77.png')
        predictions = model.predict(image)
        print(np.argmax(predictions))

    if num == '1':
        result = dict()
        for opt in optimizers:
            test(opt)

        #результаты тестирования
        for res in result:
            print("%s: %s" % (res, result[res]))