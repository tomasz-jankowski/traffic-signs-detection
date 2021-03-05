# Import required libraries
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Level 2 - display information about errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import datasets, layers, models

# Data needed for training process
MAIN_PATH = os.path.dirname(os.path.realpath(__file__))
DATADIR = f'{MAIN_PATH}\\random_data'
CATEGORIES = ["20kmh", "30kmh", "50kmh", "60kmh", "70kmh", "80kmh", "100kmh", "120kmh", "droga_z_pierwszenstwem",
              "dzikie_zwierzeta", "gololedz", "koniec_80kmh", "koniec_zakazow", "koniec_zakazu_wyprzedzania",
              "koniec_zakazu_wyprzedzania_ciezarowe", "nakaz_lewo", "nakaz_lewo_prosto", "nakaz_na_lewo",
              "nakaz_na_prawo", "nakaz_prawo", "nakaz_prawo_prosto", "nakaz_prosto", "ostre_zakrety",
              "ostry_zakret_lewo", "ostry_zakret_prawo", "pierwszenstwo_przejazdu", "przejscie_dla_pieszych",
              "roboty_drogowe", "ruch_okrezny", "sliska_jezdnia", "stop", "swiatla", "ustap_pierwszenstwa", "uwaga",
              "uwaga_dzieci", "uwaga_rower", "wyboje", "zakaz_ciezarowe", "zakaz_ruchu", "zakaz_wjazdu",
              "zakaz_wyprzedzania", "zakaz_wyprzedzania_ciezarowe", "zwezenie_prawo"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        break
    break

training_data = []

# Prepare data as a list of OpenCV elements
def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

# Next steps in data preparing process
create_training_data()

print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, 100, 100, 3))

X = np.array(X)
y = np.array(y)

# Model specification
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(100, 100, 3)))  # , initializer='he'))#, input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(Dropout(0.1))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(43))  # , activation="softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Start of training process
history = model.fit(X, y, epochs=30, callbacks=[callback], validation_split=0.25)

# Save generated model
keras_model_path = f'{MAIN_PATH}\\model'
model.save(keras_model_path)
