# Import required libraries
import os
import tkinter
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from tabulate import tabulate

tkinter.Tk().withdraw()

# Level 2 - display information about errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print('------------------------------------')
print()
print('Wybierz sposób wprowadzenia pliku:')
print()
print('[1] Okno dialogowe')
print('[2] Ścieżka')
print('[3] Wyjście z programu')
print()
print('------------------------------------')

# Input image directory
choice = 0
while not int(choice) in range(1, 4):
    choice = input("Wybrana opcja: ")

if choice == '1':
    path = askopenfilename()
    if not path is None:
        print(path)
    else:
        exit()
elif choice == '2':
    path = input('Podaj scieżkę do obrazu: ')
else:
    exit()

# Source folder path
MAIN_PATH = os.path.dirname(os.path.realpath(__file__))

# Find and use trained NN model
keras_model_path = f'{MAIN_PATH}\\model_32_64_128_256_256_plaska_30epok_vs025_2_nowe'
model = tf.keras.models.load_model(keras_model_path)

# Load image data as OpenCV element
test_image = cv2.imread(path)

# Expand dimensions to network requirements
test_image = np.expand_dims(test_image, axis=0)

# Get predictions and list it
try:
    results = model.predict(test_image)
except:
    print('Podano nieprawdiłową ścieżkę lub typ pliku!')
    exit()
results = results[0].tolist()

# Normalization
results[:] = [i / 100 for i in results]
results[:] = [round(i, 4) for i in results]

# First table column
labels = ["20kmh", "30kmh", "50kmh", "60kmh", "70kmh", "80kmh", "100kmh", "120kmh", "droga_z_pierwszenstwem",
          "dzikie_zwierzeta", "gololedz", "koniec_80kmh", "koniec_zakazow", "koniec_zakazu_wyprzedzania",
          "koniec_zakazu_wyprzedzania_ciezarowe", "nakaz_lewo", "nakaz_lewo_prosto", "nakaz_na_lewo", "nakaz_na_prawo",
          "nakaz_prawo", "nakaz_prawo_prosto", "nakaz_prosto",
          "ostre_zakrety", "ostry_zakret_lewo", "ostry_zakret_prawo", "pierwszenstwo_przejazdu",
          "przejscie_dla_pieszych", "roboty_drogowe", "ruch_okrezny", "sliska_jezdnia", "stop", "swiatla",
          "ustap_pierwszenstwa", "uwaga", "uwaga_dzieci", "uwaga_rower", "wyboje", "zakaz_ciezarowe", "zakaz_ruchu",
          "zakaz_wjazdu", "zakaz_wyprzedzania", "zakaz_wyprzedzania_ciezarowe", "zwezenie_prawo"]

# Table to show
table = []

for i in range(len(labels)):
    table.append([labels[i], results[i]])

table.sort(key=lambda x: x[1])

# Write down results
print()
print(tabulate(table,
               headers=['Klasyfikacja', 'Prawdopodobienstwo'], tablefmt='grid'))

best_match = labels[results.index(max(results))]
best_match_probability = max(results)
best_match_index = results.index(max(results))

print(
    f"\n Najlepsze dopasowanie: {best_match} \n Pewność: {best_match_probability * 100:.2f} % \n Indeks: {best_match_index}")
