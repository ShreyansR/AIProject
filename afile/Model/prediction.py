import tkinter
import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from tkinter import Menu
from PIL import ImageTk,Image
import cv2
import PIL.Image, PIL.ImageTk

import pygame
import pygame.camera


import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')                         # Use only for MAC OS
import matplotlib.pyplot as pyplot
from scipy.misc import toimage
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

StartWindow = tkinter.Tk()
StartWindow.title("ObjectAI- AN AI PROJECT PROTOTYPE")
b1name = "nowBack.gif"
bg_image = tkinter.PhotoImage(file=b1name)
w = bg_image.width()
h = bg_image.height()
StartWindow.geometry("1013x568")
cv = tkinter.Canvas(width=w, height=h)
cv.pack(side='top', fill='both', expand='yes')
cv.create_image(0, 0, image=bg_image, anchor='nw')

menubar = Menu(StartWindow)


filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", )
filemenu.add_command(label="New", )
filemenu.add_command(label="Save", )
filemenu.add_separator()
filemenu.add_command(label="Exit", command=StartWindow.quit)
menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo")
editmenu.add_command(label="Redo")
editmenu.add_command(label="Preferences")
menubar.add_cascade(label="Edit", menu=editmenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About")
menubar.add_cascade(label="Help", menu=helpmenu)

StartWindow.config(menu=menubar)




ftypes = [
    ("Image Files", "*.jpg; *.gif"),
     ("JPEG", '*.jpg'),
    ("GIF", '*.gif'),
    ('All', '*')
]


def chooseFile():
    StartWindow.sourceFile = filedialog.askopenfilename(parent=StartWindow, initialdir="/",
                                                        title='Please select a file', filetypes=ftypes)
    filename= StartWindow.sourceFile


##this is the previous prediction.py integrated as a event for the button
    def show_imgs(X):
        pyplot.figure(1)
        k = 0
        for i in range(0, 4):
            for j in range(0, 4):
                pyplot.subplot2grid((4, 4), (i, j))
                pyplot.imshow(toimage(X[k]))
                k = k + 1
        # show the plot
        pyplot.show()




    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # mean-std normalization
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    #show_imgs(x_test[:16])

    # Load trained CNN model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('model.h5')


    image = load_img(filename, target_size=(32, 32))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    indices = np.argmax(model.predict(image))

    print("The detected object in the image was a", labels[indices])
    objectD = labels[indices]
    messagebox.showinfo('ObjectAI- Result', 'The Object detected is: ' + objectD)




    #print('saving output to output.jpg')
    #indices = indices[0]
    #indices_img = image.array_to_img(indices)
    #indices_img.save('output.jpg')


#def takePic():
#    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
 #   if not cap.isOpened():
  ## while True:
    #    ret, frame = cap.read()
     ##  cv2.imshow('Input', frame)

       ##if c == 27:
        #    break

    #cap.release()
    #cv2.destroyAllWindows()




b_chooseFile = tkinter.Button(StartWindow, text ="Upload Image",bg="black",fg="white", width = 18, height = 1, command = chooseFile)
b_chooseFile.place(x = 439,y = 310, anchor='nw')
#b_chooseFile.width = 300
#b_chooseFile.pack(side='left', padx=200, pady=5, anchor='nw')

b_capture= tkinter.Button(StartWindow, text ="Capture",bg="black",fg="white", width = 18, height = 1)
b_capture.place(x = 439,y = 350, anchor='nw')




StartWindow.mainloop()
