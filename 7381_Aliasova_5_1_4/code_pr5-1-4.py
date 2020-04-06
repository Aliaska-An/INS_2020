import numpy as np
import csv
from keras.models import Model
from keras.layers import Input, Dense


def outCSV(file_name, fields, mode):
    outfile = open(file_name, 'w')
    out = csv.writer(outfile, delimiter=',')
    if mode == 1:
        for item in fields:
            out.writerow(item)
    else:
        out.writerows(map(lambda x: [x], fields))
    outfile.close()


def getData(nrow):
    data = np.zeros((nrow, 6))
    targets = np.zeros(nrow)
    for i in range(nrow):
        x = 7 * np.random.random_sample() + 3
        e = 0.03 * np.random.random_sample()
        data[i, :] = (x ** 2 + e, np.sin(x / 2) + e, np.cos(2 * x) + e, -x + e, np.absolute(x) + e, (x ** 3) / 4 + e)
        targets[i] = x - 3 + e
    return data, targets


train_data, train_targets = getData(150)
test_data, test_targets = getData(50)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

input = Input(shape=(6,))

# Encoder
encoded = Dense(30, activation='relu')(input)
encoded = Dense(30, activation='relu')(input)
encoded = Dense(15, activation='relu')(input)
encoded = Dense(3, activation='relu')(encoded)


# Decoder
decoded = Dense(60, activation='relu', name='d_layer_1')(encoded)
decoded = Dense(60, activation='relu', name='d_layer_2')(decoded)
decoded = Dense(6, name="d_layer_3")(decoded)


# Predictor
regression = Dense(64, activation='relu')(encoded)
regression = Dense(32, activation='relu')(regression)
regression = Dense(64, activation='relu')(regression)
regression = Dense(64, activation='relu')(regression)
regression = Dense(1, )(regression)


# Модели
encoder = Model(input, encoded)
autoencoder = Model(input, decoded)
encoder = Model(input, encoded)

dec_input = Input(shape=(3,))
decoder = autoencoder.get_layer('d_layer_1')(dec_input)
decoder = autoencoder.get_layer('d_layer_2')(decoder)
decoder = autoencoder.get_layer('d_layer_3')(decoder)
decoder = Model(dec_input, decoder)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train_data, train_data,
                epochs=120,
                batch_size=5,
                shuffle=True,
                validation_data=(test_data, test_data))

encoded_data = encoder.predict(test_data)
decoded_data = decoder.predict(encoded_data)

regr_model = Model(input, regression)
regr_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
regr_h = regr_model.fit(train_data, train_targets,
                        epochs=60,
                        batch_size=5,
                        verbose=1,
                        validation_data=(test_data, test_targets))


predicted_data =  regr_model.predict(test_data)


# Сохранение моделей в формате h5
decoder.save('decoder.h5')
encoder.save('encoder.h5')
regr_model.save('regressor.h5')


# Вывод
outCSV("train_data.csv", np.round(train_data, 3), 1)
outCSV("test_data.csv", np.round(test_data, 3), 1)
outCSV("train_targets.csv", np.round(train_targets, 3), 0)
outCSV("test_targets.csv", np.round(test_targets, 3), 0)

outCSV("encoded_data.csv", np.round(encoded_data, 3), 1)
outCSV("decoded_data.csv", np.round(decoded_data, 3), 1)
outCSV("regression_data.csv", np.round(np.column_stack((test_targets, predicted_data[:, 0])), 3), 1)

