import numpy as np
import mouse
import time
import win32gui
import win32api
from random import randrange
from win32con import *
import urllib.request
import pyautogui
import cv2
from PIL import Image
import glob
import shutil
import csv
import webbrowser
import os
import subprocess
import keyboard
# q

while True:  # making a loop

    if keyboard.is_pressed('b'):

        time.sleep(0.5)

        x, y = 1890, 600  # Go to stream
        mouse.move(x, y)

        time.sleep(0.5)

        mouse.click()  # Start stream

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

    if keyboard.is_pressed('n'):

        time.sleep(0.5)

        x, y = 1820, 200  # Go to next
        mouse.move(x, y)

        time.sleep(0.5)

        mouse.click()  # Next

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

    if keyboard.is_pressed('a'):

        time.sleep(0.5)

        x, y = 1050, 80  # Go to abort
        mouse.move(x, y)

        time.sleep(0.5)

        mouse.click()  # Abort

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

    if keyboard.is_pressed('p'):

        time.sleep(0.5)

        x, y = 360, 665  # Go to play
        mouse.move(x, y)

        mouse.click()  # start playing

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

    if keyboard.is_pressed('r'):

        time.sleep(0.5)

        x, y = 360, 665  # Go to pause
        mouse.move(x, y)

        mouse.click()  # pause

        time.sleep(0.5)

        x, y = 380, 665  # Go to start
        mouse.move(x, y)

        mouse.click()  # set to start

        time.sleep(0.5)

        x, y = 360, 665  # Go to play
        mouse.move(x, y)

        time.sleep(0.5)

        mouse.click()  # save

        time.sleep(0.5)

        mouse.click()  # save

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

    if keyboard.is_pressed('q'):

        time.sleep(0.5)
        x, y = 640, 1050  # Go to command ready
        mouse.move(x, y)
        time.sleep(0.5)
        mouse.click()  # Make command ready

        break