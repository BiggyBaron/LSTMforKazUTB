#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Импортируем библиотеку для взятия из под корня
from math import sqrt

# Импортируем библиотеку для функции соединения данных в таблице numpy
from numpy import concatenate

# Импортируем библиотеку для экспорта графиков
from matplotlib import pyplot

# Импортируем библиотеку pandas для работы с данными
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Импортируем библиотеку для статистического анализа данных Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Импортируем библиотеку Keras для создания нейросети
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Фнукция для считывания данных из файла data.csv
def read_data_from_csv():
    # Открываем файл csv и счтываем данные
    dataset = read_csv('./data/data.csv',  index_col="date", sep="\t")

    # Определяем заголовки и индекс заголовок
    dataset.columns = ['USD', 'RUB', 'BRENT']
    dataset.index.name = 'date'

    # Возвращаем данные
    return dataset


# Функция для рисования графиков, принимает в виде переменной список данных
def plot_values(dataset):

    # Определяем данные для графика
    values = dataset.values

    # Определяем что именно рисовать
    groups = [0, 1, 2]
    i = 1

    # Рисуем каждый график
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1

    # Показываем график
    pyplot.show()


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Функция для нормализации и оптимизации данных
# В виде входящих данных берет массив Pandas
def reframe_data(dataset):
    # Обозначает для себя данные для нормализации
    values = dataset.values
    # Создает энкодер для нормализации
    encoder = LabelEncoder()
    # Определяет какие именно данные нужны для нормализации
    values[:, 2] = encoder.fit_transform(values[:, 2])
    # Переводит данные в формат Float
    values = values.astype('float32')
    # Нормализует данные между 0 и 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # Переводит данные из серии цифр в формат тренировки с тренером
    reframed = series_to_supervised(scaled, 1, 1)
    # Убирает ненужные колонки для анализа
    reframed.drop(reframed.columns[[4, 5]], axis=1, inplace=True)
    # Возвращает нормализованные данные и scaler для последующего использования системой
    return reframed, scaler


def split_train_test(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_days = 500
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, test_X, test_y


def design_ANN(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=500, batch_size=144, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    return model


def evaluate(train_X, train_y, test_X, test_y, model, scaler):
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    yhat = 513.745153152*yhat
    test_y = 513.745153152*test_y
    pyplot.plot(yhat, label='Предугаданный')
    pyplot.plot(test_y, label='Настоящий')
    pyplot.legend()
    pyplot.show()

# Основная функция программы
def main():

    # Скачиваем данные
    dataset = read_data_from_csv()

    # Рисуем график
    plot_values(dataset)

    # reframed, scaler = reframe_data(dataset)
    # train_X, train_y, test_X, test_y = split_train_test(reframed)
    # model = design_ANN(train_X, train_y, test_X, test_y)
    # evaluate(train_X, train_y, test_X, test_y, model, scaler)


if __name__ == "__main__":
    main()
