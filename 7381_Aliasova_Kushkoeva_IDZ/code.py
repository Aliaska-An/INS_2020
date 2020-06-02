import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

EPOCHS = 500
BATCHSIZE = 8

optimizers = [optimizers.Adam(),
              optimizers.RMSprop()]


# Загрузка данных
def load_data():
    dataframe = pd.read_csv("zoo.csv")
    zoo = dataframe.values
    # Деление данных для обучения от целей
    x_data = zoo[:, 1:-1].astype(float)
    y_data = zoo[:, -1:]
    # Кодирование целей
    onehot_encoder = OneHotEncoder()
    y_data = onehot_encoder.fit_transform(y_data).toarray()
    # Нормализация данных
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[12])],
                                          remainder='passthrough')
    x_data = np.array(columnTransformer.fit_transform(x_data))
    # Деление на тренировачный и тестовый датасеты
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2,
                                                        random_state=42, stratify=y_data)
    return train_x, test_x, train_y, test_y


# Создание модели
def build_model():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam',  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Создание модели для тестирования с разными оптимизаторами
def build_model(opt):
    model = Sequential()
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer=opt,  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Тестирование модели
def train_model():
    model = build_model()
    # Обучение сети
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy',
                         mode='max', verbose=1, save_best_only=True)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                        epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[es, mc])
    create_graphics(history)
    test_loss, test_acc1 = model.evaluate(test_x, test_y)
    print("Accuracy: %.3f" % (test_acc1))
    print("Loss: %.3f" % (test_loss))

    best_model = load_model('best_model.h5')
    test_loss, test_acc2 = best_model.evaluate(test_x, test_y, verbose=0)
    print("Best accuracy: %.3f" % (test_acc2))
    print("Loss: %.3f" % (test_loss))

    if(test_acc1 == test_acc2):
        return model
    return best_model


# Тестирование модели с разными отпимизаторами
def test(opt):
    # Вывод настроек метода
    opt_config = opt.get_config()
    print("Researching with optimizer ")
    print(opt_config)

    model = build_model(opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint(filepath='best_model.hdf5', monitor='val_accuracy',
                         mode='max', verbose=1, save_best_only=True)
    # Обучение сети
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                        epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[es, mc])
    create_graphics(history)
    test_loss, test_acc1 = model.evaluate(test_x, test_y)
    result["%s" % (opt_config)] = test_acc1
    print("Accuracy: %.3f" % (test_acc1))
    print("Loss: %.3f" % (test_loss))

    best_model = load_model('best_model.hdf5')
    test_loss, test_acc2 = best_model.evaluate(test_x, test_y, verbose=0)
    print("Best accuracy: %.3f" % (test_acc2))
    print("Loss: %.3f" % (test_loss))

    if(test_acc1 == test_acc2):
        return model
    return best_model


# Создание графиков
def create_graphics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    print(len(loss))
    # График потерь
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    # График точности
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = load_data()
    num = '2'
    if (num == '1'):
        train_model()
    if (num == '2'):
        result = dict()
        for opt in optimizers:
            test(opt)
        # Результаты тестирования
        for res in result:
            print("%s: %s" % (res, result[res]))