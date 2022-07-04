from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
from tensorflow.keras.layers import LSTM, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

def get_1DCNN(x_train, y_train, x_test, y_test, epochs=50):
    model = Sequential()

    model.add(Conv1D(filters=256, kernel_size=70, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=50, activation='relu'))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=25, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))



    model.add(LSTM(units=128, activation='tanh',return_sequences=True))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(256, activation='relu', activity_regularizer=regularizers.L2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', activity_regularizer=regularizers.L2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    callback = [EarlyStopping(monitor='val_loss', patience=20),
                ModelCheckpoint(filepath='1DCNN_best_model.h5', monitor='val_loss', save_best_only=True)]

    # callback = [TensorBoard]

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=epochs,
                        callbacks=callback,
                        validation_data=(x_test, y_test))

    return history, model
def get_1DCNN_final(x_train, y_train, epochs=50):
    model = Sequential()

    model.add(Conv1D(filters=256, kernel_size=70, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=50, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=25, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Dropout(0.5))


    model.add(LSTM(units=128, activation='tanh',return_sequences=True))
    model.add(Dropout(0.5))


    model.add(GlobalAveragePooling1D())

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.00001,beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    callback = [EarlyStopping(monitor='loss', patience=20),
                ModelCheckpoint(filepath='1DCNN_best_model.h5', monitor='loss', save_best_only=True)]

    # callback = [TensorBoard]

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        callbacks=callback)

    return history, model


def Input_preprocess(X_train, y_train):

    X_train, X_test, y_train,y_test = train_test_split(np.expand_dims(X_train, axis=-1), y_train, stratify=y_train,test_size=0.1)
    y_train = to_categorical(y_train, 2, dtype='int8')
    y_true = y_test
    y_test = to_categorical(y_test, 2, dtype='int8')
    return X_train,X_test,y_train,y_test,y_true
