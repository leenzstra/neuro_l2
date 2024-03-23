import pandas as pd
import numpy as np
import net
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def prepare_data(df, column_name, size):
    prices = df[column_name].to_numpy()

    xs = []
    ys = []

    for i in range(len(prices)-size-1):
        xs.append(prices[i:size+i])
        ys.append(prices[size+i])

    return np.array(xs), np.array(ys).reshape(-1, 1)


def main():
    datafile = "Nvidia.csv"

    arch = [5, 10, 10, 1]
    activations = ['relu', 'relu', 'relu', 'relu']

    epochs = 100
    learning_rate = 0.01
    accuracy = 10 ** -6

    scale = 100

    df = pd.read_csv(datafile, sep=',')

    # подготовка данных
    xs, ys = prepare_data(df, 'close', arch[0])
    xs /= scale
    ys /= scale

    xs_train, xs_test, ys_train, ys_test = train_test_split(
        xs, ys, test_size=0.2)

    # обучение
    model = net.NN(arch, activations)
    history = model.train(xs_train, ys_train, epochs, learning_rate, accuracy)

    # тест/прогноз
    for x, y in zip(xs_test, ys_test):
        y_predicted = model.predict(x)
        print("x = {}; y = {}; y_predicted = {}".format(x, y, y_predicted))

    # график
    plt.plot(history['loss'])
    plt.show()


if __name__ == '__main__':
    main()
