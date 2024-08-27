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

def random_coords(x,y,sigma2):
    randx = randrange(sigma2-int(sigma2/2))
    x_rand = x + randx
    randy = randrange(sigma2-int(sigma2/2))
    y_rand = y + randy
    return(str(x_rand),str(y_rand))

def rt(time,sigma):
    randt = randrange(sigma)
    t_rand = time + sigma
    return(t_rand)

def search():

    try:
        while True:
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
    except KeyboardInterrupt:
        print('\n')

def read_csv_header(filename, column_idx, var_type, header_lines):
    with open(filename, encoding="utf8") as f:
        reader = csv.reader(f)
        if header_lines != 0:
            for h in range(0,header_lines):
                header = next(reader)
        vals = []
        for row in reader:
            if var_type == 'string':
                val = row[column_idx]
            if var_type == 'integer':
                val = int(row[column_idx])
            if var_type == 'float':
                if row[column_idx] == '':
                    val = -9999.0
                else:
                    val = float(row[column_idx])
            vals.append(val)
    return vals

mode = 'cull2'

if mode == 'search':

    search()

if mode == 'cull2':

    follower_file = glob.glob('C:/Users/Jeremy/Desktop/IGFollow*')[-1]
    shutil.copy(follower_file, 'C:/Users/Jeremy/Desktop/IG/follower_file.csv')

    file = 'C:/Users/Jeremy/Desktop/IG/follower_file.csv'
    followers = np.array(read_csv_header(file, 1, 'string', 1))
    #followers = np.genfromtxt(file, delimiter=',', encoding="utf8")

    for i in range(0,10):
        print(followers[i])
        #chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
        chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe '
    #    command = 'C:/"Program Files"/Google/Chrome/Application/chrome.exe https://www.instagram.com/'+followers[i]+'/'
        #webbrowser.get(chrome_path).open('https://www.instagram.com/'+followers[i]+'/')
    #    os.system(command)
        #x, y = 1730, 85  # Go to export to extensions
        #mouse.move(x, y)
        p = subprocess.Popen([chrome_path, 'https://www.instagram.com/'+followers[i]+'/following/'], close_fds=True)

        time.sleep(10.0)

        image = pyautogui.screenshot()

        image = cv2.cvtColor(np.array(image),
                         cv2.COLOR_RGB2BGR)

        # writing it to the disk using opencv
        cv2.imwrite("C:/Users/Jeremy/Desktop/screenshot/image1.png", image)
        im = Image.open("C:/Users/Jeremy/Desktop/screenshot/image1.png")
        im_crop = im.crop((1200, 485+8, 1257, 542+8))
        im_crop.save("C:/Users/Jeremy/Desktop/screenshot/image2.png", quality=95)
        im = cv2.imread("C:/Users/Jeremy/Desktop/screenshot/image2.png")
        mean_val_test = 101 < np.mean(im[im != 255]) < 103

        print(mean_val_test)

        if mean_val_test != True:

            time.sleep(2.0)

            x, y = 1875, 130 # Go to X
            mouse.move(x, y)

            time.sleep(1.0)

            mouse.click()  # save

            time.sleep(1.0)

            mouse.click()  # save

            time.sleep(1.0)

            x, y = 1580, 180 # Go to following
            mouse.move(x, y)

            time.sleep(1.0)

            mouse.click()  # save

            time.sleep(1.0)

            x, y = 1236, 762 # Go to unfollow
            mouse.move(x, y)

            # Click

            time.sleep(1.0)

        os.system('taskkill /IM chrome.exe /F')

    stop

    x, y = 1730, 85 # Go to export to extensions
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # save

    time.sleep(1.0)

    x, y = 1500, 310 # Go to export options
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # export options

    time.sleep(1.0)

    x, y = 1450, 270 # Go to following
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # export options

    time.sleep(1.0)

    x, y = 1450, 340 # Go to export following
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # export following

    time.sleep(120.0)

    x, y = 1420, 415 # Go to export to csv
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()

    time.sleep(3.0)

    x, y = 1772, 80 # Go to downloads
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()

    time.sleep(3.0)

    mouse.click()

    time.sleep(3.0)

    x, y = 1750, 200  # Go to open
    mouse.move(x, y)

    mouse.click() # open

    time.sleep(3.0)

    x, y = 1000, 120 # Go to excel bar
    mouse.move(x, y)

    mouse.click()

    time.sleep(1.0)

    mouse.drag(x,y,x+1000,y,absolute=True,duration=0.3) # Split screen excel

    time.sleep(1.0)

    x, y = 990, 60 # Go to save
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # save

    time.sleep(1.0)

    x, y = 1040, 400 # Go to save as
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # save as

    time.sleep(1.0)

    x, y = 1280, 570  # Go to browse
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # browse

    time.sleep(1.0)

    x, y = 1050, 175  # Go to desktop
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # desktop

    time.sleep(1.0)

    x, y = 1710, 560  # Go to save
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # Go to profile

if mode == 'cull':

    pyautogui.position()

    mouse.get_position()

    x, y = 1310, 500 # Move to profile name
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click() # Go to profile

    x, y = 1660, 287 # Move to following
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click() # Go to following

    time.sleep(3.0)

    image = pyautogui.screenshot()

    image = cv2.cvtColor(np.array(image),
                         cv2.COLOR_RGB2BGR)

    # writing it to the disk using opencv
    cv2.imwrite("C:/Users/Jeremy/Desktop/screenshot/image1.png", image)
    im = Image.open("C:/Users/Jeremy/Desktop/screenshot/image1.png")
    im_crop = im.crop((1200, 485, 1257, 542))
    im_crop.save("C:/Users/Jeremy/Desktop/screenshot/image2.png", quality=95)
    im = cv2.imread("C:/Users/Jeremy/Desktop/screenshot/image2.png")
    mean_val_test = 101 < np.mean(im[im != 255]) < 103

    print(mean_val_test)

    time.sleep(2.0)

    x, y = 1530, 225 # Move to following
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click()  # Exit following window

    time.sleep(1.0)

    mouse.click() # Go to following options

    x, y = 1236, 762 # Go to unfollow
    mouse.move(x, y)

    time.sleep(1.0)

    #mouse.click() # Unfollow

    x, y = 1783, 996 # Move to main profile
    mouse.move(x, y)

    time.sleep(2.0)

    mouse.click() # Exit follow options

    time.sleep(2.0)

    mouse.click() # Go to main profile

    time.sleep(2.0)

    x, y = 1652, 278 # Move to following list
    mouse.move(x, y)

    time.sleep(1.0)

    mouse.click() # Go to following list

if mode == 'follow':

    for t in range(0,500):

        x, y = random_coords(1450, 300, 10)
        mouse.move(x, y) # Pic from list
        time.sleep(rt(1.0,3.0))
        mouse.click()

        time.sleep(1.0)

        # Find profile name
        x, y = random_coords(1400, 190, 10)
        mouse.move(x, y)
        for i in range(0,400):
            time.sleep(0.01)
            mouse.move(x, str(100+i))
            if win32gui.GetCursorInfo()[1] == 65567:
                break

        time.sleep(1.0)
        mouse.click() # Go to profile

        x, y = random_coords(1150, 1000, 10)
        mouse.move(x, y) # Scan for image 1
        for i in range(0,200):
            time.sleep(0.05)
            mouse.move(x, str(1000-i))
            if win32gui.GetCursorInfo()[1] == 65539:
                break
        mouse.move("0", "-50", False)

        time.sleep(rt(1.0,3.0))

        mouse.click() # Open image 1

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1167, 566, 10)
        mouse.move(x, y) # Position center image 1

        time.sleep(rt(1.0,3.0))

        mouse.double_click(button='left') # Like image 1

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1862, 132, 10)
        mouse.move(x, y) # Move to X

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click X

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1435, 995, 10)
        mouse.move(x, y) # Scan for image 2
        for i in range(0,200):
            time.sleep(0.05)
            mouse.move(x, str(995-i))
            if win32gui.GetCursorInfo()[1] == 65539:
                break
        mouse.move("0", "-50", False)

        time.sleep(rt(1.0,3.0))

        mouse.click() # Open image 2

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1167, 566, 10)
        mouse.move(x, y) # Position center image 2

        time.sleep(rt(1.0,3.0))

        mouse.double_click(button='left') # Like image 2

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1862, 132, 10)
        mouse.move(x, y) # Move to X

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click X

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1534, 222, 10)
        mouse.move(x, y) # Move to follow

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click follow

        time.sleep(rt(4.0,3.0))

        x, y = random_coords(999, 125, 10)
        mouse.move(x, y) # Go to back

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click back

        time.sleep(rt(4.0,3.0))

        x, y = random_coords(1075, 995, 10)
        mouse.move(x, y) # Go to home

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click home

        time.sleep(rt(4.0,3.0))

        x, y = random_coords(1217, 995, 10)
        mouse.move(x, y) # Go to explore

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click explore

        time.sleep(rt(1.0,3.0))

        x, y = random_coords(1051, 75, 10)
        mouse.move(x, y) # Move to refresh

        time.sleep(rt(1.0,3.0))

        mouse.click() # Click refresh

        time.sleep(rt(5.0,3.0))