#Бблиотека
import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#ДАнные
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Преобразование
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Первая архитектура
model = Sequential()
model.add(Dense(10, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])

#Обучение
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

#Результат тестовых данных
test_acc = model.evaluate(x_test, y_test)
print(test_acc)

#Вторая архитектура
model2 = Sequential()
model2.add(Dense( 50, input_dim=784, activation='relu'))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=10, validation_split=0.1)

#Результат тестовых данных второй архитектуры
test_acc = model2.evaluate(x_test, y_test)
print(test_acc)

#Архитектура три
model3 = Sequential()
model3.add(Dense(50, input_dim=784, activation='relu'))
model3.add(Dense(50, activation='relu'))
model3.add(Dense(10, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=10, validation_split=0.1)

#Результат тестовых даных
test_acc = model3.evaluate(x_test, y_test)
print(test_acc)

#Сверточная нейроная сеть
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[:,:,:,np.newaxis] / 255.0
x_test = x_test[:,:,:,np.newaxis] / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Архитектура четвертая
model4 = Sequential()
model4.add(Conv2D(filters=64, kernel_size=2, padding='same',
activation='relu', input_shape=(28,28, 1)))
model4.add(MaxPooling2D(pool_size=2))
model4.add(Flatten())
model4.add(Dense(10, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
model4.fit(x_train, y_train, epochs=10, validation_split=0.1)

test_acc = model4.evaluate(x_test, y_test)
print(test_acc)

import numpy as np
import random

# Функция активации: логистический сигмоид
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации: сигмоид
def sigmoid_derivative(x):
    return x * (1 - x)

# Параметры нейронной сети
input_size = 2  # Количество входов
hidden_size = 4  # Количество нейронов в скрытом слое
output_size = 2  # Количество выходов
learning_rate = 0.35

# Инициализация весов случайными значениями в диапазоне от -0.3 до 0.3
np.random.seed(42)  # Для воспроизводимости
weights_input_hidden = np.random.uniform(-0.3, 0.3, (input_size, hidden_size))
weights_hidden_output = np.random.uniform(-0.3, 0.3, (hidden_size, output_size))

# Входной вектор
X = np.array([[0.1], [-0.1]])
# Эталонный выход
Y = np.array([[0.5], [-0.5]])

for epoch in range(10000):  # Количество эпох
    hidden_input = np.dot(X.T, weights_input_hidden)  # Входы в скрытый слой
    hidden_output = sigmoid(hidden_input)  # Выходы скрытого слоя

    output_input = np.dot(hidden_output, weights_hidden_output)  # Входы в выходной слой
    output_output = sigmoid(output_input)  # Выходы сети

error = Y - output_output  # Ошибка на выходе
if epoch % 1000 == 0:  # Печать ошибки каждые 1000 эпох
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    # Обратное распространение ошибки
d_output = error * sigmoid_derivative(output_output)  # Градиент для выходного слоя
error_hidden = d_output.dot(weights_hidden_output.T)  # Ошибка в скрытом слое
d_hidden = error_hidden * sigmoid_derivative(hidden_output)  # Градиент для скрытого слоя

print("Весовая матрица между входным слоем и скрытым слоем:")
print(weights_input_hidden)

print("Весовая матрица между скрытым слоем и выходным слоем:")
print(weights_hidden_output)

print("Финальный выход нейросети:")
print(output_output)