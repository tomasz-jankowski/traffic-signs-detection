# Import required libraries
import argparse
import os
import shutil
import time
import tkinter
from tkinter.filedialog import askopenfilename

import cv2
import imutils
import numpy as np

tkinter.Tk().withdraw()

# Level 2 - display information about errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

# Source folder path
MAIN_PATH = os.path.dirname(os.path.realpath(__file__))

shutil.rmtree(f'{MAIN_PATH}\\temp\\')
os.makedirs('temp')


def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=2, minSize=(50, 50)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


CATEGORIES = ["20kmh", "30kmh", "50kmh", "60kmh", "70kmh", "80kmh", "100kmh", "120kmh", "droga_z_pierwszenstwem",
              "dzikie_zwierzeta", "gololedz", "koniec_80kmh", "koniec_zakazow", "koniec_zakazu_wyprzedzania",
              "koniec_zakazu_wyprzedzania_ciezarowe", "nakaz_lewo", "nakaz_lewo_prosto", "nakaz_na_lewo",
              "nakaz_na_prawo", "nakaz_prawo", "nakaz_prawo_prosto", "nakaz_prosto",
              "ostre_zakrety", "ostry_zakret_lewo", "ostry_zakret_prawo", "pierwszenstwo_przejazdu",
              "przejscie_dla_pieszych", "roboty_drogowe", "ruch_okrezny", "sliska_jezdnia", "stop", "swiatla",
              "ustap_pierwszenstwa", "uwaga", "uwaga_dzieci", "uwaga_rower", "wyboje", "zakaz_ciezarowe", "zakaz_ruchu",
              "zakaz_wjazdu", "zakaz_wyprzedzania", "zakaz_wyprzedzania_ciezarowe", "zwezenie_prawo"]

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--size", type=str, default="(100, 100)",
                help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.6,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# Parameters needed
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 4
ROI_SIZE = eval(args["size"])
ROI_SIZE2 = (50, 50)
INPUT_SIZE = (100, 100)

print('------------------------------------')
print()
print('Wybierz sposób wprowadzenia pliku:')
print()
print('[1] Okno dialogowe')
print('[2] Ścieżka')
print('[3] Wyjście z programu')
print()
print('------------------------------------')

# Input image path
choice = 0
while not int(choice) in range(1, 4):
    choice = input("Wybrana opcja: ")

if choice == '1':
    image_path = askopenfilename()
    if not image_path is None:
        print(image_path)
    else:
        exit()
elif choice == '2':
    path = input('Podaj scieżkę do obrazu: ')
else:
    exit()

print("[INFO] loading network...")
model_path = f'{MAIN_PATH}\\model_32_64_128_256_256_plaska_30epok_vs025_2_nowe'
model = load_model(model_path)
# load the input image from disk, resize it such that it has the proper size
orig = cv2.imread(image_path)
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
# Create two lists, one for ROIs and one for their locations
rois = []
locs = []
# time how long it takes to loop over the image pyramid layers and sliding window locations
start = time.time()

tmp = 0
# Prepare lists of ROIs and locs basing on image pyramid
for image in pyramid:
    scale = W / float(image.shape[1])
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        tmp = tmp + 1
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        rois.append(roi)
        locs.append((x, y, x + w, y + h))
        if args["visualize"] > 0:
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
    end - start))
# convert the ROIs to a NumPy array
rois = np.array(rois)
# Classify ROI with the use of neural network
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
    end - start))
# Decode the predictions
preds[:] = [i / 10000 for i in preds]
print(preds.size)
a = 0
labels = {}

prev_max = 0
indexes = []
tmp1 = 0
conf = 0.02

# Search for ROIs with highest results
for pred in preds:
    index_base = np.where(pred == max(pred))
    tmp1 = 0
    if index_base not in indexes:
        indexes.append(index_base)
        tmp1 = 1
    index = index_base[0][0]
    max_value = max(pred)
    if max_value >= conf:
        box = locs[a]
        label = CATEGORIES[index]
        print(label)
        print(max_value)
        L = labels.get(label, [])
        L.append((box, max_value))
        labels[label] = L
        prev_max = max_value
    a += 1

b = 0
# Show results as a green labeled boxes in original image
for label in labels.keys():
    clone = orig.copy()
    b += 1
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    temp_path = f'{MAIN_PATH}\\temp\\{str(b + 30)}.jpg'
    print(temp_path)
    cv2.imwrite(temp_path, clone)
    boxes = np.array([p[0] for p in labels[label]])
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        res = list(labels.keys())[0]
        cv2.putText(clone, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    print(label)
    output_path = f'{MAIN_PATH}\\output.jpg'
    cv2.imwrite(output_path, clone)
