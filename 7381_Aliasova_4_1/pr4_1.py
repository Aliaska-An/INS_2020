import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def naive_relu(x):
    return np.maximum(x, 0)

def naive_sigmoid(x):
    return 1/(1+np.exp(-x))

def task(a, b, c):
    return (a and b) or (a and c)


def get_layers(weights):
    layers = []
    for i in range(len(weights)-1):
        layers.append(naive_relu)
    layers.append(naive_sigmoid)
    return layers

# Функция, в которой все операции реализованы как поэлементные операции над тензорами
def element_operation(result, weights):
    data = result.copy()
    layers = get_layers(weights)
    for i in range(len(weights)):
        step = np.zeros((len(data), len(weights[i][1])))
        for j in range(len(data)):
            for k in range(len(weights[i][1])):
                sum = 0
                for n in range(len(data[j])):
                    sum += (data[j][n] * weights[i][0][n][k])
                step[j][k] = layers[i](sum + weights[i][1][k])
        data = step
    return data

# Функция, в которой все операции реализованы с использованием операций над тензорами из NumPy
def tensor_operation(result, weights):
    data = result.copy()
    layers = get_layers(weights)
    for i in range(len(weights)):
        data = layers[i](np.dot(data, weights[i][0]) + weights[i][1])
    return data


def print_result(dataset, model):
    weights = []
    for l in model.layers:
        weights.append(l.get_weights())
    print("Модель")
    print(model.predict(dataset))
    print("Прогонка через датасета через 1ю функцию")
    print(element_operation(dataset, weights))
    print("Прогонка через датасета через 2ю функцию")
    print(tensor_operation(dataset, weights))


dataset = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]])

# Инициализация модели
model = Sequential()
model.add(Dense(5, activation = 'relu', input_shape = (3,)))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

operation_res = np.zeros((8,1))
for y in dataset:
    operation_res[y] = task(y[0], y[1], y[2])

print("Прогонка через НЕ обученную модель:")
print_result(dataset, model)

print("Прогонка обученную модель:")
# Обучаем подель
model.fit(dataset, operation_res, epochs = 35, batch_size = 2)
print_result(dataset, model)