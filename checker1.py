import numpy as np
import webbrowser
from PIL import ImageGrab
import time
import threading
from emnist import extract_training_samples
from sklearn.model_selection import train_test_split
import cv2
import pickle
import pytesseract
#import tesseract
import configparser
from PIL import Image
import requests
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def make_prediction(image_path):
    # Load the image from the URL
    #response = session.get(image_url, stream=True)
    #response.raw.decode_content = True
    img = cv2.imread(image_path)
    #img = Image.open(response.raw)
    np_frame = np.array(img)
    np_frame = np_frame[:, :, 0]
    # Match the input shape for the model
    image = np.array([np_frame / 255])

    fig, ax = plt.subplots()
    ax.imshow(image.reshape(28, 28))
    #plt.show()

    # Make the prediction
    predictions = model.predict(image)
    print(predictions)
    stop
    prediction = np.argmax(predictions[0])
    return prediction

def solve_captcha(image_path):
    captcha_image = Image.open(image_path)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Jeremy\AppData\Local\Tesseract-OCR\tesseract.exe'
    captcha_text = pytesseract.image_to_string(captcha_image)
    return captcha_text

def p(img, letter):
    A = img.load()
    B = letter.load()
    mx = 1000000
    max_x = 0
    x = 0
    for x in range(img.size[0] - letter.size[0]):
        _sum = 0
        for i in range(letter.size[0]):
            for j in range(letter.size[1]):
                _sum = _sum + abs(A[x+i, j][0] - B[i, j][0])
        if _sum < mx :
            mx = _sum
            max_x = x
    return mx, max_x

session = requests.Session()

try_new = False

os.environ['TF_CPP_IN_LOG_LEVEL'] = '2'

#(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
images, labels = extract_training_samples('letters')
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.33)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

compile_flag = True

if compile_flag:

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    with open('modelo.pkl', 'wb') as f: pickle.dump(model, f)

else:

    with open('modelo.pkl', 'rb') as f:
        model = pickle.load(f)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

prediction = make_prediction("C:/Users/Jeremy/Desktop/w_test.png")

print(prediction)

stop

if try_new:

    url = 'https://service2.diplo.de/rktermin/extern/appointment_showMonth.do?locationCode=amst&realmId=1113&categoryId=2324'

    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s &'

    webbrowser.get(chrome_path).open(url)

    time.sleep(7.0)

    # Capture a specific region (left, top, right, bottom)
    screenshot = ImageGrab.grab()

    # Save the screenshot to a file
    screenshot.save("C:/Users/Jeremy/Desktop/screenshot.png")

    # Close the screenshot
    screenshot.close()

img = cv2.imread("C:/Users/Jeremy/Desktop/screenshot.png")

y, h, x, w = 365, 60, 65, 370

crop_img = img[y:y+h, x:x+w]

cv2.imwrite("C:/Users/Jeremy/Desktop/screenshot2.png", crop_img)

captcha_solution = solve_captcha("C:/Users/Jeremy/Desktop/screenshot2.png")


