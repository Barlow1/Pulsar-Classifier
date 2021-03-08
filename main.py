# Christian Barlow
# Pulsar-Classifier

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_compiled_model():
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    sequential_model.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
    return sequential_model


def pulsar_map(mean, sd, ek, skew, meanDMSNR, sdDMSNR, ekDMSNR, skewDMSNR, target_class):
    return {"Mean of the integrated profile": mean, "Standard deviation of the integrated profile": sd,
            "Excess kurtosis of the integrated profile": ek, "Skewness of the integrated profile": skew,
            "Mean of the DM-SNR curve": meanDMSNR, "Standard deviation of the DM-SNR curve": sdDMSNR,
            "Excess kurtosis of the DM-SNR curve": ekDMSNR, "Skewness of the DM-SNR curve": skewDMSNR,
            "target_class": target_class
            }


def plot(X, index):
    plot_x = []
    for value in X.values:
        plot_x.append(value[index])
    return plot_x


def __main__(csv_path="pulsar_stars.csv"):
    LABEL_COLUMN = 'target_class'

    data = pd.read_csv(csv_path)

    y = data.target_class
    X = data.drop(LABEL_COLUMN, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=.5)

    x_train = tf.convert_to_tensor(X_train.values)
    x_test = tf.convert_to_tensor(X_test.values)
    x_val = tf.convert_to_tensor(X_val.values)
    y_train = tf.convert_to_tensor(y_train.values)
    y_test = tf.convert_to_tensor(y_test.values)
    y_val = tf.convert_to_tensor(y_val.values)

    model = get_compiled_model()

    history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val))

    print("Max Accuracy For Validation", max(history.history['accuracy']))

    print("Min Loss For Validation", min(history.history['loss']))

    x_plot = plot(X, 0)  # mean of integrated profile
    y_plot = plot(X, 1)  # standard deviation of integrated profile

    # summarize history for loss
    loss = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for accuracy
    accuracy = plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    scatter = plt.figure(0)
    plt.scatter(x_plot, y_plot, c=y)
    plt.xlabel('Mean of the integrated profile')
    plt.ylabel('Standard deviation of the integrated profile')
    plt.legend(['1', '0'], loc='upper left')

    test_predict = model.predict(x_test)
    result = model.evaluate(x_test, y_test)

    print("Test Validation: ", result)

    plt.show()


__main__()
